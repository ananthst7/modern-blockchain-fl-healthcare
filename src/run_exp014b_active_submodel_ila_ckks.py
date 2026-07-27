"""
EXP-014B: ILA-selected active-submodel HE-Assisted Multi-Krum.

This experiment keeps ILA true to its name by making the ILA-selected tensors the active trainable submodel.
The aggregation-server role does not use plaintext selected client tensors for
robust selection or aggregation. Multi-Krum selection uses HE-derived selected-
tensor distance scores, and the selected encrypted tensors are homomorphically
averaged. The decrypted HE aggregate is inserted into the global model.

Scope note: this is a single-process research prototype. Python memory contains
client states because client, server, trusted-authority and evaluator roles are
simulated in one script. The server role is still written so that it consumes
only encrypted selected tensors, metadata, distance scores, hashes and sample
counts, not plaintext selected client tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from models.efficientnet import build_efficientnet_b0
from federated.feddyn import initialize_h_states, update_h_state
from federated.server import evaluate_global_model
from aggregation.attacks import sign_flip_attack
from encryption.adaptive_tenseal_ckks import (
    create_ckks_context,
    encrypt_selected_state,
    encrypted_weighted_average,
    decrypt_selected_state,
)
from encryption.adaptive_metrics import (
    compute_adaptive_encryption_metrics,
    aggregate_adaptive_metrics,
)
from encryption.ila_selector import (
    compute_ila_scores,
    compute_ila_privacy_coverage,
    aggregate_ila_privacy_metrics,
    compute_independent_coverage_metrics,
    aggregate_independent_coverage_metrics,
)
from blockchain.poa_ledger import (
    PoALedger,
    compute_rewards,
    make_aggregation_transaction,
    make_client_update_transaction,
    make_reward_transaction,
    sha256_hex,
    tamper_copy,
)

DEFAULT_CLIENTS_ROOT = Path("data/processed/covid_clients_noniid")
DEFAULT_TEST_DIR = Path("data/processed/covid_binary/test")
DEFAULT_OUTPUT_DIR = Path("results/exp014b_active_submodel_ila_ckks")

NUM_CLASSES = 2
IMG_SIZE = 224
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=4)


def get_client_dirs(clients_root: Path, num_clients: int) -> List[Path]:
    folders = sorted([p for p in clients_root.glob("client_*") if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"No client folders found under {clients_root}")
    if len(folders) >= num_clients:
        return folders[:num_clients]
    return [folders[i % len(folders)] for i in range(num_clients)]


def estimate_full_model_communication_cost_mb(state_dict: Dict[str, torch.Tensor], num_clients: int) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    return ((total_bytes * num_clients) / (1024 ** 2)) * 2.0


def is_trainable_tensor_key(key: str) -> bool:
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False
    return key.endswith("weight") or key.endswith("bias")


def get_client_dataloader(client_dir: Path, batch_size: int) -> Tuple[DataLoader, int]:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root=str(client_dir), transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(ds)


def state_l2_distance(model: nn.Module, global_state: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    reg = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in global_state:
            reg = reg + torch.sum((param - global_state[name].to(device)) ** 2)
    return reg


def linear_correction(model: nn.Module, h_state: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    correction = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in h_state:
            correction = correction + torch.sum(param * h_state[name].to(device))
    return correction


def train_feddyn_ila_client(
    global_model: nn.Module,
    client_dir: Path,
    device: torch.device,
    h_state: Dict[str, torch.Tensor],
    alpha: float,
    local_epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], int, float, Dict[str, float], Dict[str, float]]:
    """Client-side FedDyn training plus Fisher/gradient-variance collection."""

    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True)
    model.load_state_dict(global_model.state_dict())
    model.to(device)
    model.train()

    global_state_device = {
        k: v.detach().clone().to(device)
        for k, v in global_model.state_dict().items()
        if torch.is_floating_point(v)
    }

    loader, dataset_size = get_client_dataloader(client_dir, batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    fisher_accumulator: Dict[str, float] = defaultdict(float)
    grad_norm_values: Dict[str, List[float]] = defaultdict(list)

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        progress = tqdm(loader, desc=f"FedDyn+ILA training {client_dir.name}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            reg_loss = (alpha / 2.0) * state_l2_distance(model, global_state_device, device)
            dyn_loss = linear_correction(model, h_state, device)
            loss = ce_loss + reg_loss - dyn_loss
            loss.backward()

            for name, param in named_params.items():
                if param.grad is None or not is_trainable_tensor_key(name):
                    continue
                grad = param.grad.detach()
                fisher_accumulator[name] += torch.mean(grad.pow(2)).item()
                grad_norm_values[name].append(torch.norm(grad).item())

            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

    fisher_scores = {k: v / max(total_batches, 1) for k, v in fisher_accumulator.items()}
    gradvar_scores: Dict[str, float] = {}
    for key, values in grad_norm_values.items():
        if len(values) <= 1:
            gradvar_scores[key] = 0.0
        else:
            grad_tensor = torch.tensor(values, dtype=torch.float32)
            gradvar_scores[key] = torch.var(grad_tensor, unbiased=False).item()

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    avg_loss = total_loss / max(total_batches, 1)
    return state, dataset_size, avg_loss, fisher_scores, gradvar_scores


def select_ila_keys_from_metadata(
    client_score_rows_by_client: List[List[Dict[str, Any]]],
    max_selected_bytes: int,
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    """Server-side key selection using only client-provided ILA metadata rows."""

    combined: Dict[str, Dict[str, Any]] = {}
    for rows in client_score_rows_by_client:
        for row in rows:
            key = row["key"]
            if key not in combined:
                combined[key] = {
                    "key": key,
                    "ila_score": 0.0,
                    "value_density": 0.0,
                    "tensor_bytes": int(row["tensor_bytes"]),
                    "num_params": int(row["num_params"]),
                    "update_norm": 0.0,
                    "fisher_score": 0.0,
                    "gradient_variance_score": 0.0,
                }
            combined[key]["ila_score"] += float(row.get("ila_score", 0.0))
            combined[key]["value_density"] += float(row.get("value_density", 0.0))
            combined[key]["update_norm"] += float(row.get("update_norm", 0.0))
            combined[key]["fisher_score"] += float(row.get("fisher_score", 0.0))
            combined[key]["gradient_variance_score"] += float(row.get("gradient_variance_score", 0.0))

    ranked = sorted(combined.values(), key=lambda x: x["value_density"], reverse=True)
    selected_keys: List[str] = []
    selected_bytes = 0
    for row in ranked:
        tensor_bytes = int(row["tensor_bytes"])
        if selected_bytes + tensor_bytes > max_selected_bytes:
            continue
        selected_keys.append(row["key"])
        selected_bytes += tensor_bytes

    if not selected_keys:
        raise ValueError("No ILA-CKKS tensors selected under byte budget.")
    return selected_keys, selected_bytes, ranked


def decrypt_ckks_value(value: Any) -> List[float]:
    out = value.decrypt()
    if isinstance(out, list):
        return [float(v) for v in out]
    return [float(out)]


def trusted_authority_he_distance(encrypted_a: Any, encrypted_b: Any) -> Tuple[float, str]:
    """
    Derive a pairwise squared distance from encrypted selected tensors.

    Preferred: decrypt only encrypted scalar sum of squared differences.
    Fallback: decrypt encrypted squared-difference vector and sum it in the
    trusted-authority role. Last resort: decrypt selected vectors only inside
    trusted authority. The mode is recorded in JSON for transparent reporting.
    """
    try:
        diff = encrypted_a - encrypted_b
        squared = diff * diff
        summed = squared.sum()
        dec = decrypt_ckks_value(summed)
        return max(0.0, float(dec[0])), "he_square_sum_scalar_decryption"
    except Exception:
        try:
            diff = encrypted_a - encrypted_b
            squared = diff * diff
            dec_vec = decrypt_ckks_value(squared)
            return max(0.0, float(sum(dec_vec))), "he_square_vector_decryption_sum"
        except Exception:
            va = decrypt_ckks_value(encrypted_a)
            vb = decrypt_ckks_value(encrypted_b)
            dist = sum((a - b) ** 2 for a, b in zip(va, vb))
            return max(0.0, float(dist)), "trusted_authority_plain_selected_vector_fallback"


def compute_he_distance_matrix(encrypted_vectors: List[Any]) -> Tuple[List[List[float]], Dict[str, Any]]:
    n = len(encrypted_vectors)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    modes: Dict[str, int] = {}
    start = time.perf_counter()

    for i in range(n):
        for j in range(i + 1, n):
            dist, mode = trusted_authority_he_distance(encrypted_vectors[i], encrypted_vectors[j])
            matrix[i][j] = dist
            matrix[j][i] = dist
            modes[mode] = modes.get(mode, 0) + 1

    return matrix, {
        "he_distance_time_sec": time.perf_counter() - start,
        "distance_decryption_modes": modes,
        "num_distance_pairs": n * (n - 1) // 2,
        "distance_visibility": "Server receives distance scores and selected indices, not plaintext selected client tensors.",
    }


def multikrum_from_distance_matrix(
    distance_matrix: List[List[float]],
    num_malicious: int,
    num_selected: int,
) -> Tuple[List[int], List[Tuple[float, int]]]:
    n = len(distance_matrix)
    f = num_malicious
    if n <= 2 * f + 2:
        print("Warning: Multi-Krum condition n > 2f + 2 not strictly satisfied.")

    scores: List[Tuple[float, int]] = []
    for i in range(n):
        distances = [distance_matrix[i][j] for j in range(n) if j != i]
        distances.sort()
        score = sum(distances[: max(1, n - f - 2)])
        scores.append((float(score), i))
    scores.sort(key=lambda x: x[0])
    return [idx for _, idx in scores[:num_selected]], scores


def merge_he_trainable_with_selected_buffers(
    global_state: Dict[str, torch.Tensor],
    decrypted_selected_state: Dict[str, torch.Tensor],
    selected_keys: Sequence[str],
    selected_client_states: Sequence[Dict[str, torch.Tensor]],
    selected_client_sizes: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """
    Build the complete global state for the next round.

    HE-decrypted tensors are used for every ILA-selected trainable tensor.
    Non-selected floating buffers, especially BatchNorm running_mean and
    running_var, are averaged from the same Multi-Krum-selected clients. This
    avoids loading HE-aggregated trainable weights with stale BatchNorm state
    from the previous global model, which can collapse EfficientNet accuracy.

    Non-floating buffers such as num_batches_tracked are copied from the first
    selected client.
    """

    merged = {k: v.detach().cpu().clone() for k, v in global_state.items()}
    selected_key_set = set(selected_keys)

    # 1) Insert the functional HE aggregate for selected trainable tensors.
    for key in selected_key_set:
        if key in decrypted_selected_state:
            merged[key] = decrypted_selected_state[key].detach().cpu().clone()

    # 2) Synchronize non-selected state/buffers from the same robust clients.
    total_samples = float(sum(selected_client_sizes))
    if total_samples <= 0:
        raise ValueError("Selected client sample count must be positive.")

    for key in list(merged.keys()):
        # HE-selected tensors were already inserted above.
        if key in selected_key_set:
            continue

        # Floating buffers, for example BatchNorm running_mean/running_var,
        # must track the selected clients used for the model update.
        if torch.is_floating_point(merged[key]):
            buffer_avg = torch.zeros_like(merged[key].detach().cpu())
            found_any = False

            for client_state, size in zip(selected_client_states, selected_client_sizes):
                if key in client_state and torch.is_floating_point(client_state[key]):
                    buffer_avg += client_state[key].detach().cpu() * (float(size) / total_samples)
                    found_any = True

            if found_any:
                merged[key] = buffer_avg

        # Non-floating buffers are not meaningful to average. Copy from the
        # first selected client to keep model state internally consistent.
        else:
            if selected_client_states and key in selected_client_states[0]:
                value = selected_client_states[0][key]
                if isinstance(value, torch.Tensor):
                    merged[key] = value.detach().cpu().clone()
                else:
                    merged[key] = value

    return merged


def create_round_block(
    ledger: PoALedger,
    validators: List[str],
    round_idx: int,
    client_losses: List[float],
    client_sizes: List[int],
    selected_indices: List[int],
    selected_key_count: int,
    metrics: Dict[str, Any],
    algorithm: str,
) -> float:
    clients = [f"hospital_{i}" for i in range(1, len(client_sizes) + 1)]
    selected_clients = [clients[i] for i in selected_indices]
    sample_counts = {client: int(client_sizes[i]) for i, client in enumerate(clients)}
    transactions = []

    for idx, client in enumerate(clients):
        update_hash = sha256_hex({"round": round_idx, "client": client, "kind": "client_update_hash_pointer", "algorithm": algorithm})
        encrypted_update_hash = sha256_hex({"round": round_idx, "client": client, "plain_update_hash": update_hash, "encryption": "CKKS selected ILA tensors"})
        loss = float(client_losses[idx])
        quality_proxy = round(1.0 / (1.0 + max(loss, 0.0)), 6)
        transactions.append(
            make_client_update_transaction(
                round_number=round_idx,
                client_id=client,
                update_hash=update_hash,
                encrypted_update_hash=encrypted_update_hash,
                sample_count=sample_counts[client],
                local_metrics={"loss": loss, "quality_proxy": quality_proxy},
                selected_by_aggregator=idx in selected_indices,
            )
        )

    aggregation_hash = sha256_hex({"round": round_idx, "selected_clients": selected_clients, "operation": "HE-assisted Multi-Krum + CKKS selected-tensor weighted average", "algorithm": algorithm})
    global_model_hash = sha256_hex({"round": round_idx, "accuracy": metrics.get("accuracy"), "f1": metrics.get("f1"), "note": "global model after functional HE aggregation"})
    transactions.append(
        make_aggregation_transaction(
            round_number=round_idx,
            edge_server_id="edge_server_cluster_1",
            algorithm=algorithm,
            selected_clients=selected_clients,
            aggregation_hash=aggregation_hash,
            global_model_hash=global_model_hash,
            global_metrics={"accuracy": metrics.get("accuracy"), "f1": metrics.get("f1")},
        )
    )

    local_quality = {client: round(1.0 / (1.0 + max(float(client_losses[i]), 0.0)), 6) for i, client in enumerate(clients)}
    rewards = compute_rewards(clients=clients, sample_counts=sample_counts, local_accuracies=local_quality, selected_clients=selected_clients)
    transactions.append(make_reward_transaction(round_number=round_idx, edge_server_id="edge_server_cluster_1", rewards=rewards))

    validator = validators[(round_idx - 1) % len(validators)]
    start = time.perf_counter()
    block = ledger.add_block(
        transactions=transactions,
        validator=validator,
        metadata={
            "round": round_idx,
            "num_clients": len(clients),
            "num_selected_clients": len(selected_clients),
            "selected_clients": selected_clients,
            "selected_key_count": selected_key_count,
            "blockchain_purpose": "auditability, tamper evidence, contribution tracking, traceability",
        },
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[PoA] Round {round_idx}: block={block.index}, tx={len(transactions)}, validator={validator}, hash={block.block_hash[:12]}...")
    return elapsed_ms


def write_round_history_csv(output_dir: Path, history: List[Dict[str, Any]]) -> None:
    fields = [
        "round", "accuracy", "precision", "recall", "f1", "avg_client_loss",
        "selected_indices", "selected_key_count", "selected_plain_budget_bytes",
        "selected_total_values", "avg_update_coverage_ratio", "avg_parameter_encryption_ratio",
        "avg_privacy_coverage_ratio", "avg_residual_privacy_leakage",
        "avg_leakage_coverage_ratio", "avg_variance_coverage_ratio",
        "avg_influence_coverage_ratio", "he_distance_time_sec",
        "encrypted_aggregation_time_sec", "decryption_time_sec",
        "crypto_overhead_percent", "selective_encrypted_upload_mb",
        "round_time_sec", "poa_block_creation_time_ms",
    ]
    with (output_dir / "round_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: json.dumps(row.get(field)) if isinstance(row.get(field), (list, dict)) else row.get(field) for field in fields})


def pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def write_markdown_report(output_dir: Path, results: Dict[str, Any], blockchain_summary: Dict[str, Any]) -> None:
    history = results["history"]
    final = results["final_metrics"]
    best = results["best_metrics"]
    lines: List[str] = []
    lines.append("# EXP-014 — HE-Assisted Multi-Krum with Functional ILA-CKKS Aggregation\n")
    lines.append("## Purpose\n")
    lines.append("EXP-014 fixes the previous sidecar CKKS limitation by inserting the decrypted homomorphic selected-tensor aggregate into the global model update path.\n")
    lines.append("## Architecture\n")
    lines.append("```text")
    lines.append("EfficientNet-B0 + FedDyn + ILA selection + CKKS encrypted selected tensors")
    lines.append("+ HE-derived Multi-Krum distances + Multi-Krum robust selection")
    lines.append("+ CKKS weighted averaging + trusted aggregate decryption")
    lines.append("+ functional global selected-tensor update + PoA blockchain")
    lines.append("```\n")
    lines.append("## Final Result\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Final accuracy | {pct(final['accuracy'])} |")
    lines.append(f"| Final F1 | {pct(final['f1'])} |")
    lines.append(f"| Best accuracy | {pct(best['accuracy'])} |")
    lines.append(f"| Best F1 | {pct(best['f1'])} |")
    lines.append(f"| Total runtime | {results['total_time_sec']:.2f} s |")
    lines.append(f"| Device | {results['device']} |\n")
    lines.append("## Per-Round Results\n")
    lines.append("| Round | Accuracy | F1 | Selected clients | Selected keys | HE upload MB | Crypto overhead |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|")
    for row in history:
        lines.append(f"| {row['round']} | {pct(row['accuracy'])} | {pct(row['f1'])} | {row['selected_indices']} | {row['selected_key_count']} | {row['selective_encrypted_upload_mb']:.4f} | {row['crypto_overhead_percent']:.4f}% |")
    lines.append("\n## HE-Assisted Multi-Krum\n")
    lines.append("The aggregation-server role uses distance scores derived from encrypted selected tensors. Multi-Krum ranks clients using these scores, then selected encrypted tensors are homomorphically averaged.\n")
    lines.append("| Round | Selected indices | Distance mode counts | HE distance time |")
    lines.append("|---:|---|---|---:|")
    for row in history:
        lines.append(f"| {row['round']} | {row['selected_indices']} | `{json.dumps(row.get('distance_decryption_modes', {}))}` | {row.get('he_distance_time_sec', 0.0):.4f} s |")
    lines.append("\n## ILA-CKKS Coverage Metrics\n")
    lines.append("| Round | PER | ICR | PCR | RPL | LCR | VCR | InfCR |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in history:
        lines.append(f"| {row['round']} | {pct(row['avg_parameter_encryption_ratio'])} | {pct(row['avg_update_coverage_ratio'])} | {pct(row['avg_privacy_coverage_ratio'])} | {pct(row['avg_residual_privacy_leakage'])} | {pct(row['avg_leakage_coverage_ratio'])} | {pct(row['avg_variance_coverage_ratio'])} | {pct(row['avg_influence_coverage_ratio'])} |")
    lines.append("\n## Blockchain Audit Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in ["num_blocks_including_genesis", "num_training_round_blocks", "num_transactions", "avg_block_creation_time_ms", "max_block_creation_time_ms", "verification_time_ms", "ledger_size_kb", "chain_valid", "tamper_detected"]:
        lines.append(f"| {key} | {blockchain_summary.get(key)} |")
    lines.append("\n## Valid Claims\n")
    lines.append("- CKKS is used in the functional selected-tensor global update path.")
    lines.append("- Multi-Krum remains part of the functional training path.")
    lines.append("- Robust selection uses HE-derived selected-tensor distance scores.")
    lines.append("- Selected encrypted tensors are homomorphically averaged.")
    lines.append("- The decrypted HE aggregate is inserted into the global model.")
    lines.append("- BatchNorm/non-trainable state buffers are synchronized from the same Multi-Krum-selected clients to keep EfficientNet state consistent.")
    lines.append("- The PoA ledger records hashes, selected clients, metrics and rewards.\n")
    lines.append("## Claims to Avoid\n")
    lines.append("- Fully encrypted Multi-Krum with no revealed score.")
    lines.append("- Full-model encryption of every EfficientNet tensor.")
    lines.append("- Removal of the trusted-authority assumption.")
    lines.append("- Formal MPC-level privacy.\n")
    (output_dir / "EXP-014_HE_ASSISTED_MULTIKRUM_ILA_CKKS_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")



def train_feddyn_ila_client_active(
    global_model: nn.Module,
    client_dir: Path,
    device: torch.device,
    h_state: Dict[str, torch.Tensor],
    alpha: float,
    local_epochs: int,
    lr: float,
    batch_size: int,
    active_keys: Sequence[str] | None,
    progress_label: str = "FedDyn+Active-ILA training",
) -> Tuple[Dict[str, torch.Tensor], int, float, Dict[str, float], Dict[str, float]]:
    """
    Client-side FedDyn training for an ILA-selected active submodel.

    If active_keys is None, the function behaves like a normal full trainable
    probing pass. If active_keys is provided, only those named parameters are
    trainable. This is the key EXP-014B difference: the selected tensors are not
    merely encrypted after normal training; they define the actual trainable
    submodel.
    """

    active_key_set = set(active_keys) if active_keys is not None else None

    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True)
    model.load_state_dict(global_model.state_dict())

    if active_key_set is not None:
        for name, param in model.named_parameters():
            param.requires_grad = name in active_key_set

    model.to(device)
    model.train()

    global_state_device = {
        k: v.detach().clone().to(device)
        for k, v in global_model.state_dict().items()
        if torch.is_floating_point(v)
    }

    loader, dataset_size = get_client_dataloader(client_dir, batch_size)
    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No active trainable parameters selected for the active submodel.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    fisher_accumulator: Dict[str, float] = defaultdict(float)
    grad_norm_values: Dict[str, List[float]] = defaultdict(list)

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        progress = tqdm(loader, desc=f"{progress_label} {client_dir.name}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            # FedDyn terms are computed over the model, but gradients flow only
            # through active parameters because non-active parameters are frozen.
            reg_loss = (alpha / 2.0) * state_l2_distance(model, global_state_device, device)
            dyn_loss = linear_correction(model, h_state, device)
            loss = ce_loss + reg_loss - dyn_loss
            loss.backward()

            for name, param in named_params.items():
                if param.grad is None or not is_trainable_tensor_key(name):
                    continue
                grad = param.grad.detach()
                fisher_accumulator[name] += torch.mean(grad.pow(2)).item()
                grad_norm_values[name].append(torch.norm(grad).item())

            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

    fisher_scores = {k: v / max(total_batches, 1) for k, v in fisher_accumulator.items()}
    gradvar_scores: Dict[str, float] = {}
    for key, values in grad_norm_values.items():
        if len(values) <= 1:
            gradvar_scores[key] = 0.0
        else:
            grad_tensor = torch.tensor(values, dtype=torch.float32)
            gradvar_scores[key] = torch.var(grad_tensor, unbiased=False).item()

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    avg_loss = total_loss / max(total_batches, 1)
    return state, dataset_size, avg_loss, fisher_scores, gradvar_scores


def tensor_bytes_from_state(state: Dict[str, torch.Tensor], key: str) -> int:
    value = state[key]
    return int(value.numel() * value.element_size())


def classifier_mandatory_keys(global_state: Dict[str, torch.Tensor]) -> List[str]:
    return [
        key
        for key, value in global_state.items()
        if "classifier" in key and torch.is_floating_point(value) and is_trainable_tensor_key(key)
    ]


def select_ila_keys_with_mandatory(
    client_score_rows_by_client: List[List[Dict[str, Any]]],
    max_selected_bytes: int,
    reference_state: Dict[str, torch.Tensor],
    mandatory_keys: Sequence[str],
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    """
    Budgeted ILA selection with an explicit mandatory classifier safeguard.

    The classifier is tiny but critical for transfer learning. Keeping it
    mandatory prevents a budgeted active submodel from selecting only early
    feature tensors and then failing because the classification head is frozen.
    """

    combined: Dict[str, Dict[str, Any]] = {}
    for rows in client_score_rows_by_client:
        for row in rows:
            key = row["key"]
            if key not in reference_state or not torch.is_floating_point(reference_state[key]):
                continue
            if not is_trainable_tensor_key(key):
                continue
            if key not in combined:
                combined[key] = {
                    "key": key,
                    "ila_score": 0.0,
                    "value_density": 0.0,
                    "tensor_bytes": int(row.get("tensor_bytes", tensor_bytes_from_state(reference_state, key))),
                    "num_params": int(row.get("num_params", reference_state[key].numel())),
                    "update_norm": 0.0,
                    "fisher_score": 0.0,
                    "gradient_variance_score": 0.0,
                }
            combined[key]["ila_score"] += float(row.get("ila_score", 0.0))
            combined[key]["value_density"] += float(row.get("value_density", 0.0))
            combined[key]["update_norm"] += float(row.get("update_norm", 0.0))
            combined[key]["fisher_score"] += float(row.get("fisher_score", 0.0))
            combined[key]["gradient_variance_score"] += float(row.get("gradient_variance_score", 0.0))

    ranked = sorted(combined.values(), key=lambda x: x["value_density"], reverse=True)

    selected_keys: List[str] = []
    selected_key_set = set()
    selected_bytes = 0

    for key in mandatory_keys:
        if key in reference_state and key not in selected_key_set:
            b = tensor_bytes_from_state(reference_state, key)
            selected_keys.append(key)
            selected_key_set.add(key)
            selected_bytes += b

    if selected_bytes > max_selected_bytes:
        raise ValueError(
            f"Mandatory classifier keys need {selected_bytes} bytes, "
            f"which exceeds max_selected_bytes={max_selected_bytes}."
        )

    for row in ranked:
        key = row["key"]
        if key in selected_key_set:
            continue
        tensor_bytes = int(row["tensor_bytes"])
        if selected_bytes + tensor_bytes > max_selected_bytes:
            continue
        selected_keys.append(key)
        selected_key_set.add(key)
        selected_bytes += tensor_bytes

    if not selected_keys:
        raise ValueError("No ILA active-submodel tensors selected.")

    ranked_with_mandatory = ranked
    return selected_keys, selected_bytes, ranked_with_mandatory


def sign_flip_active_update(
    client_state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
    active_keys: Sequence[str],
    scale: float,
) -> Dict[str, torch.Tensor]:
    """
    Byzantine sign-flip attack applied only to the active trainable submodel.

    This avoids corrupting frozen tensors that are not supposed to participate
    in the active-submodel update path.
    """

    active_key_set = set(active_keys)
    attacked = {k: v.detach().cpu().clone() for k, v in client_state.items()}

    for key in active_key_set:
        if key not in attacked or key not in global_state:
            continue
        if not torch.is_floating_point(attacked[key]):
            continue

        clean_delta = attacked[key] - global_state[key].detach().cpu()
        attacked[key] = global_state[key].detach().cpu() - scale * clean_delta

    return attacked


def bootstrap_active_submodel_keys(
    global_model: nn.Module,
    client_dirs: Sequence[Path],
    h_states: Sequence[Dict[str, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    budget_bytes: int,
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    """
    Metadata-only ILA bootstrap stage.

    This stage chooses the active submodel before global training. It does not
    update the global model. The selected keys then remain fixed for the budget
    run, so every locally trained parameter is also CKKS-encrypted and HE-
    aggregated.
    """

    print(f"\n[EXP-014B] Bootstrapping ILA active submodel for budget={budget_bytes} bytes")
    old_global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    client_score_rows_by_client: List[List[Dict[str, Any]]] = []

    for client_idx, client_dir in enumerate(client_dirs):
        state, _, _, fisher_scores, gradvar_scores = train_feddyn_ila_client_active(
            global_model=global_model,
            client_dir=client_dir,
            device=device,
            h_state=h_states[client_idx],
            alpha=args.alpha,
            local_epochs=args.selection_probe_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            active_keys=None,
            progress_label="ILA bootstrap probe",
        )

        # Selection metadata is produced from the benign probe update. The
        # Byzantine attack is applied during actual FL rounds.
        score_rows = compute_ila_scores(
            client_state=state,
            global_state=old_global_state_cpu,
            fisher_scores=fisher_scores,
            gradvar_scores=gradvar_scores,
        )
        client_score_rows_by_client.append(score_rows)

    mandatory_keys = classifier_mandatory_keys(old_global_state_cpu) if args.always_include_classifier else []
    selected_keys, selected_bytes, ranked = select_ila_keys_with_mandatory(
        client_score_rows_by_client=client_score_rows_by_client,
        max_selected_bytes=budget_bytes,
        reference_state=old_global_state_cpu,
        mandatory_keys=mandatory_keys,
    )
    selected_values = sum(int(old_global_state_cpu[k].numel()) for k in selected_keys if k in old_global_state_cpu)
    total_trainable_values = sum(
        int(v.numel())
        for k, v in old_global_state_cpu.items()
        if torch.is_floating_point(v) and is_trainable_tensor_key(k)
    )
    print(
        f"[EXP-014B] Active keys={len(selected_keys)} | "
        f"bytes={selected_bytes} | active PER={selected_values / max(1, total_trainable_values):.4f}"
    )
    return selected_keys, selected_bytes, ranked


def write_exp014b_budget_report(
    output_dir: Path,
    result: Dict[str, Any],
    active_keys: Sequence[str],
    ranked_ila_scores: Sequence[Dict[str, Any]],
) -> None:
    final = result["final_metrics"]
    best = result["best_metrics"]
    blockchain = result["blockchain_summary"]

    lines: List[str] = []
    lines.append("# EXP-014B — ILA-selected Active Submodel HE-assisted Multi-Krum\n")
    lines.append("## Purpose\n")
    lines.append(
        "EXP-014B tests whether selective/adaptive encryption can remain functional in the actual global-model update path "
        "without pretending that a small encrypted subset can update a fully trainable model. The core design change is that "
        "the ILA-selected tensors define the active trainable submodel. Only those tensors are locally trained, encrypted, "
        "used for HE-assisted Multi-Krum distance scoring, homomorphically averaged, decrypted by the trusted authority, and "
        "inserted back into the global model.\n"
    )
    lines.append("## Configuration\n")
    lines.append(f"- Dataset: {result['dataset']}")
    lines.append(f"- Model: {result['model']}")
    lines.append(f"- Clients: {result['num_clients']}")
    lines.append(f"- Global rounds: {result['global_rounds']}")
    lines.append(f"- Local epochs: {result['local_epochs']}")
    lines.append(f"- FedDyn alpha: {result['alpha']}")
    lines.append(f"- Budget: {result['max_selected_bytes']:,} bytes")
    lines.append(f"- Selected active keys: {len(active_keys)}")
    lines.append(f"- Selected active bytes: {result['selected_active_bytes']:,}")
    lines.append(f"- Active parameter encryption ratio: {result['active_parameter_encryption_ratio']:.6f}")
    lines.append(f"- Malicious client index: {result['malicious_client_index']}")
    lines.append(f"- Attack: sign-flip, scale {result['attack_scale']}")
    lines.append(f"- HE scheme: {result['he_scheme']} via {result['he_library']}")
    lines.append(f"- Blockchain: PoA audit ledger enabled\n")

    lines.append("## Final Result\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Final accuracy | {final['accuracy'] * 100:.2f}% |")
    lines.append(f"| Final precision | {final['precision'] * 100:.2f}% |")
    lines.append(f"| Final recall | {final['recall'] * 100:.2f}% |")
    lines.append(f"| Final F1 | {final['f1'] * 100:.2f}% |")
    lines.append(f"| Best accuracy | {best['accuracy'] * 100:.2f}% |")
    lines.append(f"| Best F1 | {best['f1'] * 100:.2f}% |")
    lines.append(f"| Final selected clients | {final['selected_indices']} |")
    lines.append(f"| Final PER | {final.get('avg_parameter_encryption_ratio', 0) * 100:.2f}% |")
    lines.append(f"| Final PCR | {final.get('avg_privacy_coverage_ratio', 0) * 100:.2f}% |")
    lines.append(f"| Final encrypted upload | {final.get('selective_encrypted_upload_mb', 0):.2f} MB |")
    lines.append(f"| Final crypto overhead | {final.get('crypto_overhead_percent', 0):.2f}% |")
    lines.append(f"| Final round time | {final.get('round_time_sec', 0):.2f} s |\n")

    lines.append("## Round-wise Result\n")
    lines.append("| Round | Accuracy | F1 | Selected Clients | Active Keys | PER | Crypto Overhead | Round Time |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in result["history"]:
        lines.append(
            f"| {row['round']} | {row['accuracy'] * 100:.2f}% | {row['f1'] * 100:.2f}% | "
            f"{row['selected_indices']} | {row['selected_key_count']} | "
            f"{row.get('avg_parameter_encryption_ratio', 0) * 100:.2f}% | "
            f"{row.get('crypto_overhead_percent', 0):.2f}% | {row.get('round_time_sec', 0):.2f}s |"
        )

    lines.append("\n## Blockchain Audit Result\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Blocks including genesis | {blockchain['num_blocks_including_genesis']} |")
    lines.append(f"| Training round blocks | {blockchain['num_training_round_blocks']} |")
    lines.append(f"| Transactions | {blockchain['num_transactions']} |")
    lines.append(f"| Average block creation time | {blockchain['avg_block_creation_time_ms']:.6f} ms |")
    lines.append(f"| Verification time | {blockchain['verification_time_ms']:.6f} ms |")
    lines.append(f"| Ledger size | {blockchain['ledger_size_kb']:.4f} KB |")
    lines.append(f"| Chain valid | {blockchain['chain_valid']} |")
    lines.append(f"| Tamper detected | {blockchain['tamper_detected']} |\n")

    lines.append("## Correct Claim\n")
    lines.append(
        "EXP-014B supports the claim that ILA-selected tensors can be used as a functional active submodel: "
        "the selected tensors are the only locally trainable parameters, and 100% of that active submodel is CKKS-encrypted, "
        "used for HE-assisted Multi-Krum, homomorphically averaged, and inserted into the global model."
    )
    lines.append("\n## Claim to Avoid\n")
    lines.append(
        "Do not claim that a small 2 MB / 4 MB / 8 MB budget updates the entire EfficientNet model. "
        "The correct framing is active-submodel learning, not full-model training under a small selective budget."
    )
    lines.append("\n## Top ILA-ranked Tensors Used During Bootstrap\n")
    lines.append("| Rank | Tensor | ILA Score | Value Density | Bytes | Params |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, row in enumerate(list(ranked_ila_scores)[:20], start=1):
        lines.append(
            f"| {rank} | `{row['key']}` | {float(row.get('ila_score', 0)):.6e} | "
            f"{float(row.get('value_density', 0)):.6e} | {int(row.get('tensor_bytes', 0))} | "
            f"{int(row.get('num_params', 0))} |"
        )

    (output_dir / "EXP-014B_ACTIVE_SUBMODEL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_single_budget(args: argparse.Namespace, budget_bytes: int) -> Dict[str, Any]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir) / f"budget_{budget_bytes}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("\n" + "=" * 90)
    print(f"EXP-014B budget run: {budget_bytes} bytes")
    print("=" * 90)
    print("Device:", device)
    print("Creating CKKS context...")
    ckks_context = create_ckks_context(
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
        global_scale=CKKS_GLOBAL_SCALE,
    )

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    client_dirs = get_client_dirs(Path(args.clients_root), args.num_clients)
    h_states = initialize_h_states(global_model.state_dict(), args.num_clients)

    active_keys, active_bytes, bootstrap_ranked_scores = bootstrap_active_submodel_keys(
        global_model=global_model,
        client_dirs=client_dirs,
        h_states=h_states,
        args=args,
        device=device,
        budget_bytes=budget_bytes,
    )
    active_key_set = set(active_keys)

    global_state_for_counts = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    active_values = sum(int(global_state_for_counts[k].numel()) for k in active_keys)
    total_trainable_values = sum(
        int(v.numel())
        for k, v in global_state_for_counts.items()
        if torch.is_floating_point(v) and is_trainable_tensor_key(k)
    )
    active_parameter_encryption_ratio = active_values / max(1, total_trainable_values)

    validators = ["edge_validator_1", "edge_validator_2", "edge_validator_3"]
    ledger = PoALedger(
        validators=validators,
        genesis_metadata={
            "experiment_id": "EXP-014B",
            "experiment_name": "ILA-selected Active Submodel HE-assisted Multi-Krum",
            "architecture": "EfficientNet-B0 + FedDyn + ILA active-submodel selection + HE-assisted Multi-Krum + PoA blockchain",
            "seed": args.seed,
            "budget_bytes": budget_bytes,
            "data_policy": "No raw patient data, raw images, or plaintext active selected client updates are stored on-chain.",
        },
    )

    history: List[Dict[str, Any]] = []
    block_creation_times_ms: List[float] = []
    total_start = time.time()

    for round_idx in range(1, args.global_rounds + 1):
        print(f"\n===== EXP-014B Active Submodel Round {round_idx}/{args.global_rounds} | budget={budget_bytes} =====")
        round_start = time.time()

        old_global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        old_global_float_cpu = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_floating_point(v)
        }

        client_states: List[Dict[str, torch.Tensor]] = []
        client_sizes: List[int] = []
        client_losses: List[float] = []
        fisher_scores_by_client: List[Dict[str, float]] = []
        gradvar_scores_by_client: List[Dict[str, float]] = []

        for client_idx, client_dir in enumerate(client_dirs):
            state, size, loss, fisher_scores, gradvar_scores = train_feddyn_ila_client_active(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                h_state=h_states[client_idx],
                alpha=args.alpha,
                local_epochs=args.local_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                active_keys=active_keys,
                progress_label="Active-submodel FedDyn+ILA training",
            )

            if client_idx == args.malicious_client_index:
                print(f"Applying active-submodel sign-flip attack to logical_client_{client_idx + 1} ({client_dir.name})")
                state = sign_flip_active_update(
                    client_state=state,
                    global_state=old_global_state_cpu,
                    active_keys=active_keys,
                    scale=args.attack_scale,
                )

            client_states.append({k: v.detach().cpu() for k, v in state.items()})
            client_sizes.append(size)
            client_losses.append(float(loss))
            fisher_scores_by_client.append(fisher_scores)
            gradvar_scores_by_client.append(gradvar_scores)

            h_states[client_idx] = update_h_state(
                h_state=h_states[client_idx],
                client_state=state,
                global_state=old_global_float_cpu,
                alpha=args.alpha,
            )

        encrypted_vectors = []
        encryption_times: List[float] = []
        encrypted_sizes: List[int] = []
        plain_selected_sizes: List[int] = []
        size_expansion_ratios: List[float] = []
        metadata = None

        for state in client_states:
            encrypted_vector, metadata, enc_metrics = encrypt_selected_state(
                state_dict=state,
                context=ckks_context,
                selected_keys=active_keys,
            )
            encrypted_vectors.append(encrypted_vector)
            encryption_times.append(float(enc_metrics["encryption_time_sec"]))
            encrypted_sizes.append(int(enc_metrics["encrypted_size_bytes"]))
            plain_selected_sizes.append(int(enc_metrics["plain_size_bytes"]))
            size_expansion_ratios.append(float(enc_metrics["size_expansion_ratio"]))

        distance_matrix, distance_metrics = compute_he_distance_matrix(encrypted_vectors)
        selected_indices, krum_scores = multikrum_from_distance_matrix(
            distance_matrix=distance_matrix,
            num_malicious=args.num_malicious,
            num_selected=args.num_selected,
        )
        print("HE-assisted Multi-Krum selected clients:", selected_indices)

        selected_encrypted_vectors = [encrypted_vectors[i] for i in selected_indices]
        selected_client_sizes = [client_sizes[i] for i in selected_indices]
        encrypted_avg, encrypted_aggregation_time = encrypted_weighted_average(
            encrypted_vectors=selected_encrypted_vectors,
            client_sizes=selected_client_sizes,
        )
        decrypted_selected_state, decryption_time = decrypt_selected_state(
            encrypted_vector=encrypted_avg,
            metadata=metadata,
            reference_state=old_global_state_cpu,
        )

        selected_client_states_for_update = [client_states[i] for i in selected_indices]
        selected_client_sizes_for_update = [client_sizes[i] for i in selected_indices]

        new_global_state = merge_he_trainable_with_selected_buffers(
            global_state=old_global_state_cpu,
            decrypted_selected_state=decrypted_selected_state,
            selected_keys=active_keys,
            selected_client_states=selected_client_states_for_update,
            selected_client_sizes=selected_client_sizes_for_update,
        )
        global_model.load_state_dict(new_global_state)
        global_model.to(device)

        metrics = evaluate_global_model(global_model, Path(args.test_dir), device)

        fixed_keys = [
            k for k in old_global_state_cpu
            if "classifier" in k and torch.is_floating_point(old_global_state_cpu[k])
        ]
        adaptive_rows = []
        ila_rows = []
        independent_rows = []
        for idx_client, state in enumerate(client_states):
            adaptive_rows.append(compute_adaptive_encryption_metrics(
                client_state=state,
                global_state=old_global_state_cpu,
                selected_keys=active_keys,
                fixed_keys=fixed_keys,
            ))
            ila_rows.append(compute_ila_privacy_coverage(
                client_state=state,
                global_state=old_global_state_cpu,
                fisher_scores=fisher_scores_by_client[idx_client],
                gradvar_scores=gradvar_scores_by_client[idx_client],
                selected_keys=active_keys,
            ))
            independent_rows.append(compute_independent_coverage_metrics(
                client_state=state,
                global_state=old_global_state_cpu,
                fisher_scores=fisher_scores_by_client[idx_client],
                gradvar_scores=gradvar_scores_by_client[idx_client],
                selected_keys=active_keys,
            ))

        adaptive_metrics = aggregate_adaptive_metrics(adaptive_rows)
        ila_metrics = aggregate_ila_privacy_metrics(ila_rows)
        independent_metrics = aggregate_independent_coverage_metrics(independent_rows)

        total_encryption_time = sum(encryption_times)
        selective_encrypted_upload_mb = sum(encrypted_sizes) / (1024 ** 2)
        selective_plain_upload_mb = sum(plain_selected_sizes) / (1024 ** 2)
        crypto_time = (
            total_encryption_time
            + float(distance_metrics["he_distance_time_sec"])
            + float(encrypted_aggregation_time)
            + float(decryption_time)
        )
        round_time = time.time() - round_start
        crypto_overhead_percent = (crypto_time / round_time) * 100 if round_time > 0 else 0.0

        poa_block_time_ms = create_round_block(
            ledger=ledger,
            validators=validators,
            round_idx=round_idx,
            client_losses=client_losses,
            client_sizes=client_sizes,
            selected_indices=selected_indices,
            selected_key_count=len(active_keys),
            metrics=metrics,
            algorithm="FedDyn + ILA-active-submodel HE-assisted Multi-Krum",
        )
        block_creation_times_ms.append(poa_block_time_ms)

        row = {
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / max(1, len(client_losses)),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "alpha": args.alpha,
            "malicious_client_index": args.malicious_client_index,
            "attack_scale": args.attack_scale,
            "selected_indices": selected_indices,
            "num_selected_by_multikrum": len(selected_indices),
            "krum_scores": krum_scores,
            "he_distance_matrix": distance_matrix,
            **distance_metrics,
            "max_selected_bytes": budget_bytes,
            "selected_plain_budget_bytes": active_bytes,
            "selected_key_count": len(active_keys),
            "selected_keys": active_keys,
            "active_submodel_keys_fixed_from_bootstrap": True,
            "selection_strategy": "bootstrap ILA metadata; active submodel = selected trainable tensors only",
            "top_ranked_ila_scores": bootstrap_ranked_scores[:20],
            "ckks_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
            "ckks_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
            "ckks_global_scale": CKKS_GLOBAL_SCALE,
            "selected_tensor_count": len(metadata["selected_keys"]),
            "selected_total_values": metadata["total_values"],
            "functional_he_update": True,
            "functional_he_update_scope": "ILA-selected active trainable submodel; non-selected trainable tensors are frozen; floating buffers synchronized from selected clients",
            "server_plaintext_selected_update_access": False,
            "trusted_authority_outputs_decrypted": ["pairwise distance scores", "final active-submodel aggregate"],
            "known_leakage": ["pairwise distance scores", "selected client indices", "active tensor names and sizes", "final active-submodel aggregate"],
            "active_parameter_encryption_ratio": active_parameter_encryption_ratio,
            "active_trainable_values": active_values,
            "total_trainable_values": total_trainable_values,
            "avg_encryption_time_sec": total_encryption_time / max(1, len(encryption_times)),
            "total_encryption_time_sec": total_encryption_time,
            "encrypted_aggregation_time_sec": float(encrypted_aggregation_time),
            "decryption_time_sec": float(decryption_time),
            "crypto_time_sec": crypto_time,
            "crypto_overhead_percent": crypto_overhead_percent,
            "avg_plain_selected_size_bytes": sum(plain_selected_sizes) / max(1, len(plain_selected_sizes)),
            "avg_encrypted_selected_size_bytes": sum(encrypted_sizes) / max(1, len(encrypted_sizes)),
            "avg_size_expansion_ratio": sum(size_expansion_ratios) / max(1, len(size_expansion_ratios)),
            "selective_plain_upload_mb": selective_plain_upload_mb,
            "selective_encrypted_upload_mb": selective_encrypted_upload_mb,
            "full_model_communication_cost_mb": estimate_full_model_communication_cost_mb(global_model.state_dict(), args.num_clients),
            "round_time_sec": round_time,
            "poa_block_creation_time_ms": poa_block_time_ms,
            "adaptive_encryption_metrics_per_client": adaptive_rows,
            "ila_privacy_metrics_per_client": ila_rows,
            "independent_coverage_metrics_per_client": independent_rows,
            **adaptive_metrics,
            **ila_metrics,
            **independent_metrics,
            "avg_gradient_energy_ratio_current_field": independent_metrics.get("avg_gradient_cosine_similarity", 0.0),
            **metrics,
        }
        history.append(json_safe(row))

        print(
            f"Round {round_idx} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
            f"HE-MK selected: {selected_indices} | Active PER: {active_parameter_encryption_ratio:.4f} | "
            f"PCR: {row.get('avg_privacy_coverage_ratio', 0):.4f} | "
            f"HE distance: {distance_metrics['he_distance_time_sec']:.4f}s | "
            f"HE agg: {encrypted_aggregation_time:.4f}s | Dec: {decryption_time:.4f}s | "
            f"Overhead: {crypto_overhead_percent:.2f}% | Time: {round_time:.2f}s"
        )

    total_time = time.time() - total_start
    ledger_path = output_dir / "poa_audit_ledger.json"
    ledger.save(ledger_path)

    verify_start = time.perf_counter()
    validation = ledger.validate_chain()
    verification_time_ms = (time.perf_counter() - verify_start) * 1000
    tampered = tamper_copy(ledger)
    tamper_validation = tampered.validate_chain()

    blockchain_summary = {
        "blockchain_type": "Proof-of-Authority audit ledger",
        "consensus": "Authorized rotating validators",
        "num_validators": len(validators),
        "num_blocks_including_genesis": validation["num_blocks"],
        "num_training_round_blocks": validation["num_blocks"] - 1,
        "num_transactions": validation["num_transactions"],
        "avg_block_creation_time_ms": statistics.mean(block_creation_times_ms) if block_creation_times_ms else 0.0,
        "max_block_creation_time_ms": max(block_creation_times_ms) if block_creation_times_ms else 0.0,
        "verification_time_ms": verification_time_ms,
        "ledger_size_kb": ledger_path.stat().st_size / 1024,
        "chain_valid": validation["valid"],
        "tamper_detected": not tamper_validation["valid"],
        "data_policy": "Only hashes, metrics, selected-client metadata, active-submodel distance metadata and reward records are stored on-chain.",
    }
    save_json(output_dir / "poa_audit_summary.json", blockchain_summary)
    save_json(output_dir / "tamper_test.json", {"tamper_detected": not tamper_validation["valid"], "validation_after_tamper": tamper_validation})

    final_results = {
        "experiment": "EXP-014B",
        "method": "ILA-selected Active Submodel HE-Assisted Multi-Krum",
        "dataset": "COVID Radiography Binary",
        "setting": "Moderate Non-IID + Byzantine sign-flip attack",
        "num_clients": args.num_clients,
        "global_rounds": args.global_rounds,
        "local_epochs": args.local_epochs,
        "selection_probe_epochs": args.selection_probe_epochs,
        "learning_rate": args.lr,
        "alpha": args.alpha,
        "model": "EfficientNet-B0",
        "aggregation": "HE-assisted Multi-Krum over ILA-selected active submodel",
        "he_library": "TenSEAL",
        "he_scheme": "CKKS",
        "selection_strategy": "ILA-selected active trainable submodel",
        "max_selected_bytes": budget_bytes,
        "selected_active_bytes": active_bytes,
        "selected_active_key_count": len(active_keys),
        "active_parameter_encryption_ratio": active_parameter_encryption_ratio,
        "active_trainable_values": active_values,
        "total_trainable_values": total_trainable_values,
        "functional_he_update": True,
        "functional_he_update_scope": "Only ILA-selected tensors are trainable and HE-aggregated; non-selected trainable tensors remain frozen.",
        "blockchain_enabled": True,
        "privacy_architecture_note": "Server role does not receive plaintext active selected client updates; trusted authority decrypts distance scores and final active-submodel aggregate. Single-process prototype simulates all roles in one runtime.",
        "malicious_client_index": args.malicious_client_index,
        "attack_scale": args.attack_scale,
        "seed": args.seed,
        "device": str(device),
        "total_time_sec": total_time,
        "active_keys": active_keys,
        "history": history,
        "final_metrics": history[-1],
        "best_metrics": max(history, key=lambda row: row["accuracy"]),
        "blockchain_summary": blockchain_summary,
    }

    save_json(output_dir / f"covid_exp014b_active_submodel_budget_{budget_bytes}.json", final_results)
    write_round_history_csv(output_dir, history)

    if args.save_model:
        torch.save(global_model.state_dict(), output_dir / f"covid_exp014b_active_submodel_budget_{budget_bytes}.pth")

    write_exp014b_budget_report(output_dir, final_results, active_keys, bootstrap_ranked_scores)

    print("\n===== EXP-014B Budget Summary =====")
    print(json.dumps({
        "budget_bytes": budget_bytes,
        "active_key_count": len(active_keys),
        "active_parameter_encryption_ratio": active_parameter_encryption_ratio,
        "final_accuracy": history[-1]["accuracy"],
        "final_f1": history[-1]["f1"],
        "best_accuracy": final_results["best_metrics"]["accuracy"],
        "best_f1": final_results["best_metrics"]["f1"],
        "chain_valid": blockchain_summary["chain_valid"],
        "tamper_detected": blockchain_summary["tamper_detected"],
    }, indent=4))
    print("\nSaved EXP-014B budget outputs to:", output_dir)
    return final_results


def write_exp014b_master_summary(output_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    rows = []
    for result in results:
        final = result["final_metrics"]
        best = result["best_metrics"]
        blockchain = result["blockchain_summary"]
        rows.append({
            "budget_bytes": result["max_selected_bytes"],
            "active_key_count": result["selected_active_key_count"],
            "active_bytes": result["selected_active_bytes"],
            "active_parameter_encryption_ratio": result["active_parameter_encryption_ratio"],
            "final_accuracy": final["accuracy"],
            "final_f1": final["f1"],
            "best_accuracy": best["accuracy"],
            "best_f1": best["f1"],
            "final_encrypted_upload_mb": final.get("selective_encrypted_upload_mb", 0.0),
            "final_crypto_overhead_percent": final.get("crypto_overhead_percent", 0.0),
            "final_round_time_sec": final.get("round_time_sec", 0.0),
            "chain_valid": blockchain["chain_valid"],
            "tamper_detected": blockchain["tamper_detected"],
        })

    save_json(output_dir / "exp014b_active_submodel_summary.json", {"experiments": rows})

    with (output_dir / "exp014b_active_submodel_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    lines: List[str] = []
    lines.append("# EXP-014B Active Submodel Budget Summary\n")
    lines.append("EXP-014B evaluates whether selective/adaptive ILA-CKKS can remain functional when the selected tensors define the active trainable submodel.\n")
    lines.append("| Budget | Active Keys | Active Bytes | Active PER | Final Acc | Final F1 | Best Acc | Encrypted Upload | Crypto Overhead | Chain Valid | Tamper |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        lines.append(
            f"| {row['budget_bytes']:,} | {row['active_key_count']} | {row['active_bytes']:,} | "
            f"{row['active_parameter_encryption_ratio'] * 100:.2f}% | {row['final_accuracy'] * 100:.2f}% | "
            f"{row['final_f1'] * 100:.2f}% | {row['best_accuracy'] * 100:.2f}% | "
            f"{row['final_encrypted_upload_mb']:.2f} MB | {row['final_crypto_overhead_percent']:.2f}% | "
            f"{row['chain_valid']} | {row['tamper_detected']} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "If lower budgets preserve useful accuracy, this supports the claim that ILA can be used as a functional selective-encryption mechanism. "
        "If accuracy drops at lower budgets, that is still a valid result: it quantifies the utility cost of shrinking the active encrypted submodel."
    )
    (output_dir / "EXP-014B_ACTIVE_SUBMODEL_MASTER_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for budget in args.budgets:
        budget_args = argparse.Namespace(**vars(args))
        budget_args.max_selected_bytes = int(budget)
        results.append(run_single_budget(budget_args, int(budget)))

    write_exp014b_master_summary(output_dir, results)
    return {"experiment": "EXP-014B", "budget_results": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-014B ILA-selected active-submodel HE-assisted Multi-Krum")
    parser.add_argument("--clients-root", default=str(DEFAULT_CLIENTS_ROOT))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--global-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--selection-probe-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-malicious", type=int, default=1)
    parser.add_argument("--num-selected", type=int, default=2)
    parser.add_argument("--malicious-client-index", type=int, default=0)
    parser.add_argument("--attack-scale", type=float, default=5.0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2_000_000, 4_000_000, 8_000_000])
    parser.add_argument("--always-include-classifier", action="store_true")
    parser.add_argument("--no-always-include-classifier", dest="always_include_classifier", action="store_false")
    parser.set_defaults(always_include_classifier=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-model", dest="save_model", action="store_true")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false")
    parser.set_defaults(save_model=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
