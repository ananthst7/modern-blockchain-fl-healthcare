import time
import sys
from typing import Dict, Tuple, List

import torch
import tenseal as ts

def select_topk_update_keys(client_state, global_state, top_k=4):
    """
    Select top-k meaningful trainable tensors by update magnitude.

    Excludes BatchNorm running statistics and tracking buffers because they are
    not trainable parameters and are not ideal candidates for adaptive encryption.
    """

    scores = []

    for key, tensor in client_state.items():
        if key not in global_state:
            continue

        if not torch.is_floating_point(tensor):
            continue

        # Ignore BatchNorm/statistical buffers
        if (
            "running_mean" in key
            or "running_var" in key
            or "num_batches_tracked" in key
        ):
            continue

        # Keep only trainable-style parameter tensors
        if not (key.endswith("weight") or key.endswith("bias")):
            continue

        client_tensor = tensor.detach().cpu().float()
        global_tensor = global_state[key].detach().cpu().float()

        if client_tensor.shape != global_tensor.shape:
            continue

        delta = client_tensor - global_tensor
        update_norm = torch.norm(delta).item()
        num_params = client_tensor.numel()

        # Normalize slightly so huge layers do not dominate purely by size
        normalized_score = update_norm / (num_params ** 0.5)

        scores.append({
            "key": key,
            "update_norm": update_norm,
            "num_params": num_params,
            "score": update_norm,
        })

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    selected = scores[:top_k]

    selected_keys = [item["key"] for item in selected]

    return selected_keys, selected

def create_ckks_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
    global_scale: float = 2**40,
):
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = global_scale
    return context


def is_selected_key(key: str) -> bool:
    return "classifier" in key


def flatten_selected_state(state_dict, selected_keys):
    shapes = []
    lengths = []
    flat_parts = []

    for key in selected_keys:
        tensor = state_dict[key]

        if torch.is_floating_point(tensor):
            tensor_cpu = tensor.detach().cpu().float()
            shapes.append(tuple(tensor_cpu.shape))
            flat = tensor_cpu.flatten()
            lengths.append(flat.numel())
            flat_parts.append(flat)

    if not flat_parts:
        raise ValueError("No tensors selected for adaptive CKKS encryption.")

    flat_vector = torch.cat(flat_parts).tolist()

    metadata = {
        "selected_keys": selected_keys,
        "shapes": shapes,
        "lengths": lengths,
        "total_values": len(flat_vector),
    }

    return flat_vector, metadata


def reconstruct_selected_state(
    decrypted_vector: List[float],
    metadata: dict,
    reference_state: Dict[str, torch.Tensor],
):
    updated_state = {
        key: value.detach().cpu().clone()
        for key, value in reference_state.items()
    }

    cursor = 0

    for key, shape, length in zip(
        metadata["selected_keys"],
        metadata["shapes"],
        metadata["lengths"],
    ):
        values = decrypted_vector[cursor: cursor + length]
        tensor = torch.tensor(values, dtype=updated_state[key].dtype).reshape(shape)
        updated_state[key] = tensor
        cursor += length

    return updated_state


def encrypt_selected_state(state_dict, context, selected_keys):
    flat_vector, metadata = flatten_selected_state(state_dict, selected_keys)

    start = time.time()
    encrypted_vector = ts.ckks_vector(context, flat_vector)
    encryption_time = time.time() - start

    serialized = encrypted_vector.serialize()
    encrypted_size_bytes = len(serialized)

    plain_size_bytes = sum(
        state_dict[key].detach().cpu().numel() * state_dict[key].detach().cpu().element_size()
        for key in selected_keys
        if torch.is_floating_point(state_dict[key])
    )

    return encrypted_vector, metadata, {
        "encryption_time_sec": encryption_time,
        "encrypted_size_bytes": encrypted_size_bytes,
        "plain_size_bytes": plain_size_bytes,
        "size_expansion_ratio": encrypted_size_bytes / max(1, plain_size_bytes),
    }


def encrypted_weighted_average(encrypted_vectors, client_sizes):
    total_samples = sum(client_sizes)
    weights = [size / total_samples for size in client_sizes]

    start = time.time()

    encrypted_avg = encrypted_vectors[0] * weights[0]

    for encrypted_vector, weight in zip(encrypted_vectors[1:], weights[1:]):
        encrypted_avg += encrypted_vector * weight

    aggregation_time = time.time() - start

    return encrypted_avg, aggregation_time


def decrypt_selected_state(encrypted_vector, metadata, reference_state):
    start = time.time()
    decrypted_vector = encrypted_vector.decrypt()
    decryption_time = time.time() - start

    updated_state = reconstruct_selected_state(
        decrypted_vector=decrypted_vector,
        metadata=metadata,
        reference_state=reference_state,
    )

    return updated_state, decryption_time