"""
EXP-011: Full end-to-end integrated framework run.

Final architecture:
EfficientNet-B0 + FedDyn local training + Multi-Krum Byzantine defense
+ Information Leakage-Aware CKKS profiling + Proof-of-Authority blockchain audit.

Run from repo root:
    python src/run_full_framework_integrated.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.efficientnet import build_efficientnet_b0
from federated.feddyn import initialize_h_states, update_h_state
from federated.server import evaluate_global_model
from aggregation.attacks import sign_flip_attack
from aggregation.multikrum import multikrum
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
    select_ila_keys_under_budget,
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


CLIENTS_ROOT = Path("data/processed/covid_clients_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
DEFAULT_RESULTS_DIR = Path("results/full_framework_integrated")

NUM_CLASSES = 2
DEFAULT_NUM_CLIENTS = 4
DEFAULT_GLOBAL_ROUNDS = 5
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_LR = 1e-4
DEFAULT_ALPHA = 0.005
DEFAULT_MALICIOUS_CLIENT_INDEX = 0
DEFAULT_ATTACK_SCALE = 5.0
DEFAULT_MAX_SELECTED_BYTES = 2_000_000

CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2**40

IMG_SIZE = 224
BATCH_SIZE = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def write_scalar_history_csv(history: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = sorted({k for row in history for k, v in row.items() if is_scalar(v)})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in scalar_keys})


def get_client_dirs(clients_root: Path, num_clients: int) -> Tuple[List[Path], bool]:
    base_dirs = [clients_root / f"client_{i}" for i in range(1, 10_000) if (clients_root / f"client_{i}").exists()]
    if not base_dirs:
        raise FileNotFoundError(f"No client folders found under {clients_root}")

    reused = num_clients > len(base_dirs)
    client_dirs = [base_dirs[i % len(base_dirs)] for i in range(num_clients)]
    return client_dirs, reused


def get_client_dataloader(client_dir: Path, batch_size: int = BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(root=str(client_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return loader, len(dataset)


def is_trainable_tensor_key(key: str) -> bool:
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False
    return key.endswith("weight") or key.endswith("bias")


def state_l2_distance(model: nn.Module, global_state: Dict[str, torch.Tensor], device) -> torch.Tensor:
    reg = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in global_state:
            reg += torch.sum((param - global_state[name].to(device)) ** 2)
    return reg


def linear_correction(model: nn.Module, h_state: Dict[str, torch.Tensor], device) -> torch.Tensor:
    correction = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in h_state:
            correction += torch.sum(param * h_state[name].to(device))
    return correction


def train_feddyn_ila_client(
    global_model: nn.Module,
    client_dir: Path,
    device,
    h_state: Dict[str, torch.Tensor],
    alpha: float = DEFAULT_ALPHA,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = BATCH_SIZE,
):
    """
    FedDyn local training plus ILA metric collection.

    This combines the FedDyn local objective with the Fisher-like and gradient-variance
    statistics needed by ILA-CKKS.
    """

    client_model = copy.deepcopy(global_model).to(device)
    client_model.train()

    global_state = {
        k: v.detach().clone().to(device)
        for k, v in global_model.state_dict().items()
        if torch.is_floating_point(v)
    }

    loader, dataset_size = get_client_dataloader(client_dir, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr)

    fisher_accumulator: Dict[str, float] = defaultdict(float)
    grad_norm_values: Dict[str, List[float]] = defaultdict(list)
    named_params = {name: param for name, param in client_model.named_parameters() if param.requires_grad}

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        progress = tqdm(loader, desc=f"FedDyn-ILA training {client_dir.name}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = client_model(images)
            ce_loss = criterion(outputs, labels)
            reg_loss = (alpha / 2.0) * state_l2_distance(client_model, global_state, device)
            dyn_loss = linear_correction(client_model, h_state, device)
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

    fisher_scores = {key: value / max(total_batches, 1) for key, value in fisher_accumulator.items()}
    gradvar_scores: Dict[str, float] = {}
    for key, values in grad_norm_values.items():
        if len(values) <= 1:
            gradvar_scores[key] = 0.0
        else:
            gradvar_scores[key] = torch.var(torch.tensor(values, dtype=torch.float32), unbiased=False).item()

    state = {key: value.detach().cpu() for key, value in client_model.state_dict().items()}
    avg_loss = total_loss / max(total_batches, 1)
    return state, dataset_size, avg_loss, fisher_scores, gradvar_scores


def estimate_full_model_communication_cost_mb(state_dict: Dict[str, torch.Tensor], num_clients: int) -> float:
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)
    return upload_mb + download_mb


def choose_num_selected(num_clients: int, num_malicious: int) -> int:
    if num_clients <= 4:
        return 2
    return max(2, num_clients - 2 * num_malicious - 2)


def create_ledger(num_clients: int, seed: int) -> PoALedger:
    validators = ["edge_validator_1", "edge_validator_2", "edge_validator_3"]
    return PoALedger(
        validators=validators,
        genesis_metadata={
            "experiment_id": "EXP-011",
            "experiment_name": "Full End-to-End Integrated Framework",
            "architecture": "EfficientNet-B0 + FedDyn + Multi-Krum + ILA-CKKS + PoA blockchain",
            "seed": seed,
            "data_policy": "No raw patient data, raw images, raw model weights, or decrypted updates are stored on-chain.",
        },
    )


def add_round_to_ledger(
    ledger: PoALedger,
    round_idx: int,
    num_clients: int,
    selected_indices: List[int],
    client_sizes: List[int],
    client_losses: List[float],
    metrics: Dict[str, float],
    selected_keys: List[str],
    crypto_summary: Dict[str, Any],
    validators: List[str],
) -> float:
    clients = [f"hospital_{i}" for i in range(1, num_clients + 1)]
    selected_clients = [f"hospital_{i + 1}" for i in selected_indices]

    sample_counts = {client: int(client_sizes[i]) for i, client in enumerate(clients)}
    local_quality = {
        client: round(1.0 / (1.0 + max(float(client_losses[i]), 0.0)), 6)
        for i, client in enumerate(clients)
    }

    transactions = []
    for idx, client in enumerate(clients):
        update_hash = sha256_hex({
            "round": round_idx,
            "client": client,
            "kind": "FedDyn local update",
            "loss": client_losses[idx],
            "sample_count": client_sizes[idx],
        })
        encrypted_update_hash = sha256_hex({
            "round": round_idx,
            "client": client,
            "plain_update_hash": update_hash,
            "selected_key_count": len(selected_keys),
            "ckks": "ILA-selected CKKS ciphertext pointer",
        })

        transactions.append(
            make_client_update_transaction(
                round_number=round_idx,
                client_id=client,
                update_hash=update_hash,
                encrypted_update_hash=encrypted_update_hash,
                sample_count=sample_counts[client],
                local_metrics={"loss": float(client_losses[idx]), "quality_proxy": local_quality[client]},
                selected_by_aggregator=client in selected_clients,
            )
        )

    aggregation_hash = sha256_hex({
        "round": round_idx,
        "selected_clients": selected_clients,
        "algorithm": "FedDyn + Multi-Krum + ILA-CKKS",
        "crypto_summary": crypto_summary,
    })
    global_model_hash = sha256_hex({
        "round": round_idx,
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "note": "off-chain model checkpoint hash pointer",
    })

    transactions.append(
        make_aggregation_transaction(
            round_number=round_idx,
            edge_server_id="edge_server_cluster_1",
            algorithm="FedDyn + Multi-Krum + ILA-CKKS",
            selected_clients=selected_clients,
            aggregation_hash=aggregation_hash,
            global_model_hash=global_model_hash,
            global_metrics={"accuracy": metrics.get("accuracy"), "f1": metrics.get("f1")},
        )
    )

    rewards = compute_rewards(
        clients=clients,
        sample_counts=sample_counts,
        local_accuracies=local_quality,
        selected_clients=selected_clients,
    )
    transactions.append(
        make_reward_transaction(
            round_number=round_idx,
            edge_server_id="edge_server_cluster_1",
            rewards=rewards,
        )
    )

    validator = validators[(round_idx - 1) % len(validators)]
    start = time.perf_counter()
    ledger.add_block(
        transactions=transactions,
        validator=validator,
        metadata={
            "round": round_idx,
            "num_clients": num_clients,
            "num_selected_clients": len(selected_clients),
            "selected_clients": selected_clients,
            "selected_key_count": len(selected_keys),
            "blockchain_purpose": "auditability, tamper evidence, contribution tracking, traceability",
        },
    )
    return (time.perf_counter() - start) * 1000


def run_integrated_framework(
    output_dir: Path = DEFAULT_RESULTS_DIR,
    clients_root: Path = CLIENTS_ROOT,
    test_dir: Path = TEST_DIR,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    global_rounds: int = DEFAULT_GLOBAL_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    lr: float = DEFAULT_LR,
    alpha: float = DEFAULT_ALPHA,
    malicious_client_index: int = DEFAULT_MALICIOUS_CLIENT_INDEX,
    attack_scale: float = DEFAULT_ATTACK_SCALE,
    max_selected_bytes: int = DEFAULT_MAX_SELECTED_BYTES,
    seed: int = 7,
    enable_blockchain: bool = True,
    save_model: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Creating CKKS context...")
    ckks_context = create_ckks_context(
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
        global_scale=CKKS_GLOBAL_SCALE,
    )

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    client_dirs, reused_client_dirs = get_client_dirs(clients_root, num_clients)
    h_states = initialize_h_states(global_model.state_dict(), num_clients)
    num_selected = choose_num_selected(num_clients, num_malicious=1)

    validators = ["edge_validator_1", "edge_validator_2", "edge_validator_3"]
    ledger = create_ledger(num_clients=num_clients, seed=seed) if enable_blockchain else None
    block_creation_times_ms: List[float] = []

    history: List[Dict[str, Any]] = []
    start_total = time.time()

    for round_idx in range(1, global_rounds + 1):
        print(f"\n===== EXP-011 Full Framework Round {round_idx}/{global_rounds} =====")
        round_start = time.time()

        global_state_cpu = {key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()}
        old_global_float_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items() if torch.is_floating_point(v)}

        client_states: List[Dict[str, torch.Tensor]] = []
        client_sizes: List[int] = []
        client_losses: List[float] = []
        fisher_scores_by_client: List[Dict[str, float]] = []
        gradvar_scores_by_client: List[Dict[str, float]] = []

        for client_idx, client_dir in enumerate(client_dirs):
            state, size, loss, fisher_scores, gradvar_scores = train_feddyn_ila_client(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                h_state=h_states[client_idx],
                alpha=alpha,
                local_epochs=local_epochs,
                lr=lr,
            )

            if client_idx == malicious_client_index:
                print(f"Applying sign-flip attack to logical_client_{client_idx + 1} ({client_dir.name})")
                state = sign_flip_attack(state, scale=attack_scale)

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(float(loss))
            fisher_scores_by_client.append(fisher_scores)
            gradvar_scores_by_client.append(gradvar_scores)

            h_states[client_idx] = update_h_state(
                h_state=h_states[client_idx],
                client_state=state,
                global_state=old_global_float_state,
                alpha=alpha,
            )

        selected_keys, selected_plain_budget_bytes, ranked_ila_scores = select_ila_keys_under_budget(
            client_states=client_states,
            global_state=global_state_cpu,
            fisher_scores_by_client=fisher_scores_by_client,
            gradvar_scores_by_client=gradvar_scores_by_client,
            max_selected_bytes=max_selected_bytes,
        )

        ila_rows = [
            compute_ila_privacy_coverage(client_states[i], global_state_cpu, fisher_scores_by_client[i], gradvar_scores_by_client[i], selected_keys)
            for i in range(num_clients)
        ]
        ila_metrics = aggregate_ila_privacy_metrics(ila_rows)

        independent_rows = [
            compute_independent_coverage_metrics(client_states[i], global_state_cpu, fisher_scores_by_client[i], gradvar_scores_by_client[i], selected_keys)
            for i in range(num_clients)
        ]
        independent_metrics = aggregate_independent_coverage_metrics(independent_rows)

        fixed_keys = [key for key in global_state_cpu.keys() if "classifier" in key and torch.is_floating_point(global_state_cpu[key])]
        adaptive_rows = [
            compute_adaptive_encryption_metrics(client_state=state, global_state=global_state_cpu, selected_keys=selected_keys, fixed_keys=fixed_keys)
            for state in client_states
        ]
        adaptive_metrics = aggregate_adaptive_metrics(adaptive_rows)

        encrypted_vectors = []
        encryption_times = []
        encrypted_sizes = []
        plain_selected_sizes = []
        size_expansion_ratios = []
        metadata = None

        for idx, state in enumerate(client_states):
            encrypted_vector, metadata, enc_metrics = encrypt_selected_state(
                state_dict=state,
                context=ckks_context,
                selected_keys=selected_keys,
            )
            encrypted_vectors.append(encrypted_vector)
            encryption_times.append(float(enc_metrics["encryption_time_sec"]))
            encrypted_sizes.append(int(enc_metrics["encrypted_size_bytes"]))
            plain_selected_sizes.append(int(enc_metrics["plain_size_bytes"]))
            size_expansion_ratios.append(float(enc_metrics["size_expansion_ratio"]))

        encrypted_avg, encrypted_aggregation_time = encrypted_weighted_average(encrypted_vectors, client_sizes)
        _, decryption_time = decrypt_selected_state(encrypted_avg, metadata, global_state_cpu)

        new_global_state, selected_indices, krum_scores = multikrum(
            client_states=client_states,
            client_sizes=client_sizes,
            num_malicious=1,
            num_selected=num_selected,
        )
        print("Multi-Krum selected clients:", selected_indices)

        global_model.load_state_dict(new_global_state)
        global_model.to(device)
        metrics = evaluate_global_model(global_model, test_dir, device)

        full_model_comm_mb = estimate_full_model_communication_cost_mb(global_model.state_dict(), num_clients)
        avg_encryption_time = sum(encryption_times) / len(encryption_times)
        total_encryption_time = sum(encryption_times)
        avg_encrypted_size = sum(encrypted_sizes) / len(encrypted_sizes)
        avg_plain_selected_size = sum(plain_selected_sizes) / len(plain_selected_sizes)
        avg_size_expansion = sum(size_expansion_ratios) / len(size_expansion_ratios)
        selective_encrypted_upload_mb = sum(encrypted_sizes) / (1024 ** 2)
        selective_plain_upload_mb = sum(plain_selected_sizes) / (1024 ** 2)
        round_time = time.time() - round_start
        crypto_time = total_encryption_time + encrypted_aggregation_time + decryption_time
        crypto_overhead_percent = (crypto_time / round_time) * 100 if round_time > 0 else 0.0

        crypto_summary = {
            "selected_key_count": len(selected_keys),
            "selected_plain_budget_bytes": selected_plain_budget_bytes,
            "selective_encrypted_upload_mb": selective_encrypted_upload_mb,
            "crypto_overhead_percent": crypto_overhead_percent,
        }

        block_creation_time_ms = None
        if ledger is not None:
            block_creation_time_ms = add_round_to_ledger(
                ledger=ledger,
                round_idx=round_idx,
                num_clients=num_clients,
                selected_indices=selected_indices,
                client_sizes=client_sizes,
                client_losses=client_losses,
                metrics=metrics,
                selected_keys=selected_keys,
                crypto_summary=crypto_summary,
                validators=validators,
            )
            block_creation_times_ms.append(block_creation_time_ms)

        icr = adaptive_metrics.get("avg_update_coverage_ratio", 0.0)
        rlr = adaptive_metrics.get("avg_information_leakage_ratio", 0.0)
        rrs = adaptive_metrics.get("avg_reconstruction_risk_score", 0.0)
        per = adaptive_metrics.get("avg_parameter_encryption_ratio", 0.0)
        aeq = adaptive_metrics.get("avg_adaptive_encryption_quality", 0.0)
        pcr = ila_metrics.get("avg_privacy_coverage_ratio", 0.0)
        rpl = ila_metrics.get("avg_residual_privacy_leakage", 0.0)

        row: Dict[str, Any] = {
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "alpha": alpha,
            "malicious_client_index": malicious_client_index,
            "attack_scale": attack_scale,
            "selected_indices": selected_indices,
            "num_selected_by_multikrum": len(selected_indices),
            "krum_scores": krum_scores,
            "max_selected_bytes": max_selected_bytes,
            "selected_plain_budget_bytes": selected_plain_budget_bytes,
            "selected_key_count": len(selected_keys),
            "selected_keys": selected_keys,
            "selection_strategy": "ILA score = update_norm * fisher_score * gradient_variance",
            "top_ranked_ila_scores": ranked_ila_scores[:20],
            "ckks_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
            "ckks_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
            "ckks_global_scale": CKKS_GLOBAL_SCALE,
            "selected_tensor_count": len(metadata["selected_keys"]),
            "selected_total_values": metadata["total_values"],
            "avg_encryption_time_sec": avg_encryption_time,
            "total_encryption_time_sec": total_encryption_time,
            "encrypted_aggregation_time_sec": float(encrypted_aggregation_time),
            "decryption_time_sec": float(decryption_time),
            "crypto_time_sec": float(crypto_time),
            "crypto_overhead_percent": float(crypto_overhead_percent),
            "avg_plain_selected_size_bytes": avg_plain_selected_size,
            "avg_encrypted_selected_size_bytes": avg_encrypted_size,
            "avg_size_expansion_ratio": avg_size_expansion,
            "selective_plain_upload_mb": selective_plain_upload_mb,
            "selective_encrypted_upload_mb": selective_encrypted_upload_mb,
            "full_model_communication_cost_mb": full_model_comm_mb,
            "round_time_sec": round_time,
            "poa_block_creation_time_ms": block_creation_time_ms,
            "avg_information_coverage_ratio": icr,
            "avg_residual_leakage_risk": rlr,
            "avg_reconstruction_risk_score": rrs,
            "avg_parameter_encryption_ratio": per,
            "avg_adaptive_encryption_quality": aeq,
            "avg_privacy_coverage_ratio": pcr,
            "avg_residual_privacy_leakage": rpl,
            "avg_leakage_coverage_ratio": independent_metrics.get("avg_leakage_coverage_ratio", 0.0),
            "avg_variance_coverage_ratio": independent_metrics.get("avg_variance_coverage_ratio", 0.0),
            "avg_influence_coverage_ratio": independent_metrics.get("avg_influence_coverage_ratio", 0.0),
            "avg_gradient_energy_ratio_current_field": independent_metrics.get("avg_gradient_cosine_similarity", 0.0),
            "adaptive_encryption_metrics_per_client": adaptive_rows,
            "ila_privacy_metrics_per_client": ila_rows,
            "independent_coverage_metrics_per_client": independent_rows,
            **adaptive_metrics,
            **ila_metrics,
            **independent_metrics,
            **metrics,
        }
        history.append(json_safe(row))

        print(
            f"Round {round_idx} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
            f"Selected: {selected_indices} | ICR: {icr:.4f} | PCR: {pcr:.4f} | PER: {per:.6f} | "
            f"LCR: {independent_metrics.get('avg_leakage_coverage_ratio', 0):.4f} | "
            f"VCR: {independent_metrics.get('avg_variance_coverage_ratio', 0):.4f} | "
            f"InfCR: {independent_metrics.get('avg_influence_coverage_ratio', 0):.4f} | "
            f"Enc Upload: {selective_encrypted_upload_mb:.4f} MB | Crypto: {crypto_overhead_percent:.4f}% | "
            f"Round Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    blockchain_metrics: Optional[Dict[str, Any]] = None
    if ledger is not None:
        ledger_path = output_dir / "poa_audit_ledger.json"
        ledger.save(ledger_path)

        verify_start = time.perf_counter()
        validation = ledger.validate_chain()
        verification_time_ms = (time.perf_counter() - verify_start) * 1000
        tampered = tamper_copy(ledger)
        tamper_validation = tampered.validate_chain()

        blockchain_metrics = {
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
            "data_policy": "Only hashes, metrics, selected-client metadata, and reward records are stored on-chain.",
        }
        with (output_dir / "poa_audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(json_safe(blockchain_metrics), f, indent=4)

    final_results = {
        "experiment": "EXP-011",
        "method": "Full Framework: FedDyn + Multi-Krum + ILA-CKKS + PoA Blockchain",
        "dataset": "COVID Radiography Binary",
        "setting": "Moderate Non-IID + Byzantine sign-flip attack",
        "num_clients": num_clients,
        "client_dirs_reused_for_scalability": reused_client_dirs,
        "global_rounds": global_rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "alpha": alpha,
        "model": "EfficientNet-B0",
        "aggregation": "Multi-Krum",
        "he_library": "TenSEAL",
        "he_scheme": "CKKS",
        "selection_strategy": "Information Leakage-Aware adaptive tensor selection",
        "max_selected_bytes": max_selected_bytes,
        "blockchain_enabled": enable_blockchain,
        "blockchain_note": "Blockchain does not change model accuracy; it adds auditability, tamper evidence, and contribution tracking.",
        "malicious_client_index": malicious_client_index,
        "attack_scale": attack_scale,
        "seed": seed,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
        "best_metrics": max(history, key=lambda row: row["accuracy"]),
        "blockchain_metrics": blockchain_metrics,
    }

    with (output_dir / "full_framework_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(final_results), f, indent=4)

    write_scalar_history_csv(history, output_dir / "full_framework_round_history.csv")

    if save_model:
        torch.save(global_model.state_dict(), output_dir / "full_framework_model.pth")

    print("\nSaved full framework results to:", output_dir)
    return json_safe(final_results)


def parse_args():
    parser = argparse.ArgumentParser(description="Run EXP-011 full integrated FL + ILA-CKKS + PoA framework.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--clients-root", type=str, default=str(CLIENTS_ROOT))
    parser.add_argument("--test-dir", type=str, default=str(TEST_DIR))
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--global-rounds", type=int, default=DEFAULT_GLOBAL_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--max-selected-bytes", type=int, default=DEFAULT_MAX_SELECTED_BYTES)
    parser.add_argument("--malicious-client-index", type=int, default=DEFAULT_MALICIOUS_CLIENT_INDEX)
    parser.add_argument("--attack-scale", type=float, default=DEFAULT_ATTACK_SCALE)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-blockchain", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_integrated_framework(
        output_dir=Path(args.output_dir),
        clients_root=Path(args.clients_root),
        test_dir=Path(args.test_dir),
        num_clients=args.num_clients,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        alpha=args.alpha,
        malicious_client_index=args.malicious_client_index,
        attack_scale=args.attack_scale,
        max_selected_bytes=args.max_selected_bytes,
        seed=args.seed,
        enable_blockchain=not args.no_blockchain,
        save_model=not args.no_save_model,
    )


if __name__ == "__main__":
    main()

