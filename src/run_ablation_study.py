"""
EXP-012: Ablation study for the final framework.

Fixed setting for fair comparison:
- COVID binary classification
- Moderate Non-IID hospital split
- 4 clients
- 1 Byzantine sign-flip client
- 5 global rounds
- EfficientNet-B0

Rows:
1. FedAvg
2. FedDyn
3. FedDyn + Multi-Krum
4. FedDyn + Multi-Krum + CKKS
5. FedDyn + Multi-Krum + ILA-CKKS
6. Full framework + Blockchain

Run from repo root:
    python src/run_ablation_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from models.efficientnet import build_efficientnet_b0
from federated.client import train_client
from federated.feddyn_client import train_feddyn_client
from federated.fedavg import fedavg
from federated.feddyn import initialize_h_states, update_h_state, feddyn_aggregate
from federated.server import evaluate_global_model
from aggregation.attacks import sign_flip_attack
from aggregation.multikrum import multikrum
from encryption.tenseal_ckks import (
    create_ckks_context,
    encrypt_selected_state,
    encrypted_weighted_average,
    decrypt_selected_state,
)
from run_full_framework_integrated import (
    run_integrated_framework,
    set_seed,
    estimate_full_model_communication_cost_mb,
)


CLIENTS_ROOT = Path("data/processed/covid_clients_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/ablation")

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
ALPHA = 0.005
NUM_CLASSES = 2
MALICIOUS_CLIENT_INDEX = 0
ATTACK_SCALE = 5.0
SEED = 7

CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2**40


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def client_dirs() -> List[Path]:
    return [CLIENTS_ROOT / f"client_{i}" for i in range(1, NUM_CLIENTS + 1)]


def final_row(method: str, history: List[Dict[str, Any]], **extra) -> Dict[str, Any]:
    final = history[-1]
    best = max(history, key=lambda row: row["accuracy"])
    return {
        "method": method,
        "accuracy": final.get("accuracy"),
        "f1": final.get("f1"),
        "best_accuracy": best.get("accuracy"),
        "best_f1": best.get("f1"),
        "final_round": final.get("round"),
        "round_time_sec": final.get("round_time_sec"),
        **extra,
    }


def save_method_result(name: str, payload: Dict[str, Any]) -> None:
    path = RESULTS_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=4)


def run_fedavg_attack(device) -> Dict[str, Any]:
    print("\n========== ABLATION: FedAvg ==========")
    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        round_start = time.time()
        states, sizes, losses = [], [], []

        for idx, cdir in enumerate(client_dirs()):
            state, size, loss = train_client(global_model, cdir, device, LOCAL_EPOCHS, LR)
            state = {k: v.cpu() for k, v in state.items()}
            if idx == MALICIOUS_CLIENT_INDEX:
                print(f"Applying sign-flip attack to {cdir.name}")
                state = sign_flip_attack(state, scale=ATTACK_SCALE)
            states.append(state)
            sizes.append(size)
            losses.append(float(loss))

        avg_state = fedavg(states, sizes)
        global_model.load_state_dict(avg_state)
        global_model.to(device)
        metrics = evaluate_global_model(global_model, TEST_DIR, device)
        row = {
            "round": round_idx,
            "avg_client_loss": sum(losses) / len(losses),
            "client_losses": losses,
            "client_sizes": sizes,
            "round_time_sec": time.time() - round_start,
            "communication_cost_mb": estimate_full_model_communication_cost_mb(global_model.state_dict(), NUM_CLIENTS),
            **metrics,
        }
        history.append(row)
        print(f"FedAvg round {round_idx}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    payload = {
        "experiment": "EXP-012",
        "method": "FedAvg under Byzantine sign-flip attack",
        "setting": "Moderate Non-IID + 1 Byzantine client",
        "history": history,
        "final_metrics": history[-1],
        "total_time_sec": time.time() - start_total,
    }
    save_method_result("fedavg", payload)
    return payload


def run_feddyn_attack(device) -> Dict[str, Any]:
    print("\n========== ABLATION: FedDyn ==========")
    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    h_states = initialize_h_states(global_model.state_dict(), NUM_CLIENTS)
    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        round_start = time.time()
        old_global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items() if torch.is_floating_point(v)}
        states, sizes, losses = [], [], []

        for idx, cdir in enumerate(client_dirs()):
            state, size, loss = train_feddyn_client(global_model, cdir, device, h_states[idx], ALPHA, LOCAL_EPOCHS, LR)
            if idx == MALICIOUS_CLIENT_INDEX:
                print(f"Applying sign-flip attack to {cdir.name}")
                state = sign_flip_attack(state, scale=ATTACK_SCALE)
            states.append(state)
            sizes.append(size)
            losses.append(float(loss))
            h_states[idx] = update_h_state(h_states[idx], state, old_global_state, ALPHA)

        new_state = feddyn_aggregate(states, sizes, h_states, ALPHA)
        global_model.load_state_dict(new_state)
        global_model.to(device)
        metrics = evaluate_global_model(global_model, TEST_DIR, device)
        row = {
            "round": round_idx,
            "avg_client_loss": sum(losses) / len(losses),
            "client_losses": losses,
            "client_sizes": sizes,
            "round_time_sec": time.time() - round_start,
            "communication_cost_mb": estimate_full_model_communication_cost_mb(global_model.state_dict(), NUM_CLIENTS),
            "alpha": ALPHA,
            **metrics,
        }
        history.append(row)
        print(f"FedDyn round {round_idx}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

    payload = {
        "experiment": "EXP-012",
        "method": "FedDyn under Byzantine sign-flip attack",
        "setting": "Moderate Non-IID + 1 Byzantine client",
        "history": history,
        "final_metrics": history[-1],
        "total_time_sec": time.time() - start_total,
    }
    save_method_result("feddyn", payload)
    return payload


def run_feddyn_multikrum(device, use_ckks: bool = False) -> Dict[str, Any]:
    label = "FedDyn + Multi-Krum + CKKS" if use_ckks else "FedDyn + Multi-Krum"
    print(f"\n========== ABLATION: {label} ==========")
    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    h_states = initialize_h_states(global_model.state_dict(), NUM_CLIENTS)
    history = []
    start_total = time.time()

    ckks_context = None
    if use_ckks:
        print("Creating classifier-only CKKS context...")
        ckks_context = create_ckks_context(CKKS_POLY_MODULUS_DEGREE, CKKS_COEFF_MOD_BIT_SIZES, CKKS_GLOBAL_SCALE)

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        round_start = time.time()
        old_global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items() if torch.is_floating_point(v)}
        states, sizes, losses = [], [], []

        encryption_times = []
        encrypted_sizes = []
        plain_sizes = []
        encrypted_aggregation_time = 0.0
        decryption_time = 0.0
        encrypted_vectors = []
        metadata = None

        for idx, cdir in enumerate(client_dirs()):
            state, size, loss = train_feddyn_client(global_model, cdir, device, h_states[idx], ALPHA, LOCAL_EPOCHS, LR)
            if idx == MALICIOUS_CLIENT_INDEX:
                print(f"Applying sign-flip attack to {cdir.name}")
                state = sign_flip_attack(state, scale=ATTACK_SCALE)

            if use_ckks:
                encrypted_vector, metadata, enc_metrics = encrypt_selected_state(state, ckks_context)
                encrypted_vectors.append(encrypted_vector)
                encryption_times.append(float(enc_metrics["encryption_time_sec"]))
                encrypted_sizes.append(int(enc_metrics["encrypted_size_bytes"]))
                plain_sizes.append(int(enc_metrics["plain_size_bytes"]))

            states.append(state)
            sizes.append(size)
            losses.append(float(loss))
            h_states[idx] = update_h_state(h_states[idx], state, old_global_state, ALPHA)

        if use_ckks:
            encrypted_avg, encrypted_aggregation_time = encrypted_weighted_average(encrypted_vectors, sizes)
            global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
            _, decryption_time = decrypt_selected_state(encrypted_avg, metadata, global_state_cpu)

        new_state, selected_indices, krum_scores = multikrum(states, sizes, num_malicious=1, num_selected=2)
        global_model.load_state_dict(new_state)
        global_model.to(device)
        metrics = evaluate_global_model(global_model, TEST_DIR, device)

        round_time = time.time() - round_start
        crypto_time = sum(encryption_times) + encrypted_aggregation_time + decryption_time
        row = {
            "round": round_idx,
            "avg_client_loss": sum(losses) / len(losses),
            "client_losses": losses,
            "client_sizes": sizes,
            "selected_indices": selected_indices,
            "krum_scores": krum_scores,
            "round_time_sec": round_time,
            "communication_cost_mb": estimate_full_model_communication_cost_mb(global_model.state_dict(), NUM_CLIENTS),
            "alpha": ALPHA,
            "ckks_enabled": use_ckks,
            "selected_layer_scope": "classifier-only" if use_ckks else "none",
            "avg_encryption_time_sec": (sum(encryption_times) / len(encryption_times)) if encryption_times else 0.0,
            "encrypted_aggregation_time_sec": encrypted_aggregation_time,
            "decryption_time_sec": decryption_time,
            "crypto_overhead_percent": (crypto_time / round_time) * 100 if round_time > 0 else 0.0,
            "selective_encrypted_upload_mb": (sum(encrypted_sizes) / (1024 ** 2)) if encrypted_sizes else 0.0,
            "selective_plain_upload_mb": (sum(plain_sizes) / (1024 ** 2)) if plain_sizes else 0.0,
            **metrics,
        }
        history.append(row)
        print(f"{label} round {round_idx}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, selected={selected_indices}")

    payload = {
        "experiment": "EXP-012",
        "method": label,
        "setting": "Moderate Non-IID + 1 Byzantine client",
        "history": history,
        "final_metrics": history[-1],
        "total_time_sec": time.time() - start_total,
    }
    save_method_result("feddyn_multikrum_ckks" if use_ckks else "feddyn_multikrum", payload)
    return payload


def write_summary_csv(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS_DIR / "ablation_summary.csv"
    fieldnames = [
        "method",
        "accuracy",
        "f1",
        "best_accuracy",
        "best_f1",
        "byzantine_defense",
        "ckks",
        "ila_selection",
        "blockchain",
        "main_purpose",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Run EXP-012 ablation study.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--skip-basic", action="store_true", help="Only run final ILA/full framework rows.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rows: List[Dict[str, Any]] = []

    if not args.skip_basic:
        fedavg_result = run_fedavg_attack(device)
        rows.append(final_row(
            "FedAvg",
            fedavg_result["history"],
            byzantine_defense="No",
            ckks="No",
            ila_selection="No",
            blockchain="No",
            main_purpose="Basic FL baseline under attack",
        ))

        feddyn_result = run_feddyn_attack(device)
        rows.append(final_row(
            "FedDyn",
            feddyn_result["history"],
            byzantine_defense="No",
            ckks="No",
            ila_selection="No",
            blockchain="No",
            main_purpose="Non-IID robustness under attack",
        ))

        feddyn_mk_result = run_feddyn_multikrum(device, use_ckks=False)
        rows.append(final_row(
            "FedDyn + Multi-Krum",
            feddyn_mk_result["history"],
            byzantine_defense="Yes",
            ckks="No",
            ila_selection="No",
            blockchain="No",
            main_purpose="Byzantine robustness",
        ))

        feddyn_mk_ckks_result = run_feddyn_multikrum(device, use_ckks=True)
        rows.append(final_row(
            "FedDyn + Multi-Krum + CKKS",
            feddyn_mk_ckks_result["history"],
            byzantine_defense="Yes",
            ckks="Yes, classifier-only CKKS profiling",
            ila_selection="No",
            blockchain="No",
            main_purpose="Encrypted update transmission profiling",
        ))

    full_result = run_integrated_framework(
        output_dir=RESULTS_DIR / "full_framework_from_ablation",
        clients_root=CLIENTS_ROOT,
        test_dir=TEST_DIR,
        num_clients=NUM_CLIENTS,
        global_rounds=GLOBAL_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        alpha=ALPHA,
        malicious_client_index=MALICIOUS_CLIENT_INDEX,
        attack_scale=ATTACK_SCALE,
        max_selected_bytes=2_000_000,
        seed=args.seed,
        enable_blockchain=True,
        save_model=False,
    )

    # Blockchain is non-learning audit infrastructure. The ILA row and full-framework row
    # share the same model trajectory; the full-framework row additionally includes PoA audit logging.
    rows.append(final_row(
        "FedDyn + Multi-Krum + ILA-CKKS",
        full_result["history"],
        byzantine_defense="Yes",
        ckks="Yes",
        ila_selection="Yes",
        blockchain="No",
        main_purpose="Privacy-aware selective encryption",
        note="Accuracy shared with full run because blockchain does not alter model training.",
    ))

    rows.append(final_row(
        "Full Framework + Blockchain",
        full_result["history"],
        byzantine_defense="Yes",
        ckks="Yes",
        ila_selection="Yes",
        blockchain="Yes, PoA audit ledger",
        main_purpose="Auditable secure FL system",
        blockchain_chain_valid=full_result.get("blockchain_metrics", {}).get("chain_valid"),
        blockchain_tamper_detected=full_result.get("blockchain_metrics", {}).get("tamper_detected"),
    ))

    payload = {
        "experiment": "EXP-012",
        "experiment_name": "Ablation Study",
        "fixed_setting": {
            "dataset": "COVID Radiography Binary",
            "split": "Moderate Non-IID",
            "clients": NUM_CLIENTS,
            "malicious_clients": 1,
            "attack": "sign-flip",
            "attack_scale": ATTACK_SCALE,
            "global_rounds": GLOBAL_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "alpha": ALPHA,
            "seed": args.seed,
        },
        "summary": rows,
        "important_note": "The blockchain layer does not change classification accuracy; it adds auditability, tamper evidence, contribution tracking, and traceability.",
    }

    with (RESULTS_DIR / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=4)
    write_summary_csv(rows)

    print("\n===== EXP-012 ABLATION SUMMARY =====")
    print(json.dumps(json_safe(rows), indent=4))
    print("\nSaved ablation results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
