import time
import sys
from typing import Dict, Tuple, List

import torch
import tenseal as ts


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


def flatten_selected_state(state_dict: Dict[str, torch.Tensor]):
    selected_keys = []
    shapes = []
    lengths = []
    flat_parts = []

    for key, tensor in state_dict.items():
        if is_selected_key(key) and torch.is_floating_point(tensor):
            tensor_cpu = tensor.detach().cpu().float()
            selected_keys.append(key)
            shapes.append(tuple(tensor_cpu.shape))
            flat = tensor_cpu.flatten()
            lengths.append(flat.numel())
            flat_parts.append(flat)

    if not flat_parts:
        raise ValueError("No selected classifier tensors found for CKKS encryption.")

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


def encrypt_selected_state(state_dict, context):
    flat_vector, metadata = flatten_selected_state(state_dict)

    start = time.time()
    encrypted_vector = ts.ckks_vector(context, flat_vector)
    encryption_time = time.time() - start

    serialized = encrypted_vector.serialize()
    encrypted_size_bytes = len(serialized)

    plain_size_bytes = sum(
        tensor.detach().cpu().numel() * tensor.detach().cpu().element_size()
        for key, tensor in state_dict.items()
        if is_selected_key(key) and torch.is_floating_point(tensor)
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