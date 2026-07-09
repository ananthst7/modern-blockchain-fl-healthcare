from pathlib import Path
import json
import time

import torch

from models.efficientnet import build_efficientnet_b0
from federated.client import train_client
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
from encryption.privacy_aware_selector import get_privacy_aware_selected_keys


CLIENTS_ROOT = Path("data/processed/covid_clients_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/multikrum_privacy_aware_ckks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 2

MALICIOUS_CLIENT_INDEX = 0
ATTACK_SCALE = 5.0

MAX_SELECTED_BYTES = 2_000_000
TOP_K = None

EMA_ALPHA = 0.2
UPDATE_WEIGHT = 0.5
HISTORY_WEIGHT = 0.3
LAYER_WEIGHT = 0.2

CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2**40


def estimate_full_model_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)
    return upload_mb + download_mb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Creating CKKS context...")
    ckks_context = create_ckks_context(
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
        global_scale=CKKS_GLOBAL_SCALE,
    )

    global_model = build_efficientnet_b0(
        num_classes=NUM_CLASSES,
        pretrained=True,
    ).to(device)

    client_dirs = [
        CLIENTS_ROOT / f"client_{i}"
        for i in range(1, NUM_CLIENTS + 1)
    ]

    historical_scores = {}
    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== Privacy-Aware Multi-Krum + CKKS Round {round_idx}/{GLOBAL_ROUNDS} =====")
        round_start = time.time()

        global_state_cpu = {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        }

        client_states = []
        client_sizes = []
        client_losses = []

        for client_idx, client_dir in enumerate(client_dirs):
            state, size, loss = train_client(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                local_epochs=LOCAL_EPOCHS,
                lr=LR,
            )

            state = {key: value.cpu() for key, value in state.items()}

            if client_idx == MALICIOUS_CLIENT_INDEX:
                print(f"Applying sign-flip attack to {client_dir.name}")
                state = sign_flip_attack(state, scale=ATTACK_SCALE)

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)

        selected_keys, selected_plain_budget_bytes, historical_scores, ranked_scores = (
            get_privacy_aware_selected_keys(
                client_states=client_states,
                global_state=global_state_cpu,
                historical_scores=historical_scores,
                max_selected_bytes=MAX_SELECTED_BYTES,
                top_k=TOP_K,
                ema_alpha=EMA_ALPHA,
                update_weight=UPDATE_WEIGHT,
                history_weight=HISTORY_WEIGHT,
                layer_weight=LAYER_WEIGHT,
            )
        )

        print(f"Selected {len(selected_keys)} privacy-aware CKKS keys")
        print(f"Selected plaintext bytes: {selected_plain_budget_bytes}")
        print("Top selected keys:")
        for key in selected_keys[:10]:
            print(" -", key)

        fixed_keys = [
            key
            for key in global_state_cpu.keys()
            if "classifier" in key and torch.is_floating_point(global_state_cpu[key])
        ]

        adaptive_metric_rows = []

        for state in client_states:
            adaptive_metric_rows.append(
                compute_adaptive_encryption_metrics(
                    client_state=state,
                    global_state=global_state_cpu,
                    selected_keys=selected_keys,
                    fixed_keys=fixed_keys,
                )
            )

        adaptive_metrics = aggregate_adaptive_metrics(adaptive_metric_rows)

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
            encryption_times.append(enc_metrics["encryption_time_sec"])
            encrypted_sizes.append(enc_metrics["encrypted_size_bytes"])
            plain_selected_sizes.append(enc_metrics["plain_size_bytes"])
            size_expansion_ratios.append(enc_metrics["size_expansion_ratio"])

            print(
                f"client_{idx + 1} | "
                f"enc_time={enc_metrics['encryption_time_sec']:.4f}s | "
                f"plain={enc_metrics['plain_size_bytes']} bytes | "
                f"encrypted={enc_metrics['encrypted_size_bytes']} bytes | "
                f"expansion={enc_metrics['size_expansion_ratio']:.2f}x"
            )

        encrypted_avg, encrypted_aggregation_time = encrypted_weighted_average(
            encrypted_vectors=encrypted_vectors,
            client_sizes=client_sizes,
        )

        _, decryption_time = decrypt_selected_state(
            encrypted_vector=encrypted_avg,
            metadata=metadata,
            reference_state=global_state_cpu,
        )

        multikrum_state, selected_indices, krum_scores = multikrum(
            client_states=client_states,
            client_sizes=client_sizes,
            num_malicious=1,
            num_selected=2,
        )

        print("Multi-Krum selected clients:", selected_indices)

        global_model.load_state_dict(multikrum_state)
        global_model.to(device)

        metrics = evaluate_global_model(global_model, TEST_DIR, device)

        full_model_comm_mb = estimate_full_model_communication_cost_mb(
            global_model.state_dict(),
            NUM_CLIENTS,
        )

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

        icr = adaptive_metrics.get("avg_update_coverage_ratio", 0.0)
        rlr = adaptive_metrics.get("avg_information_leakage_ratio", 0.0)
        rrs = adaptive_metrics.get("avg_reconstruction_risk_score", 0.0)
        per = adaptive_metrics.get("avg_parameter_encryption_ratio", 0.0)
        aeq = adaptive_metrics.get("avg_adaptive_encryption_quality", 0.0)

        row = {
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "malicious_client_index": MALICIOUS_CLIENT_INDEX,
            "attack_scale": ATTACK_SCALE,
            "selected_indices": selected_indices,
            "krum_scores": krum_scores,
            "max_selected_bytes": MAX_SELECTED_BYTES,
            "top_k": TOP_K,
            "selected_plain_budget_bytes": selected_plain_budget_bytes,
            "selected_key_count": len(selected_keys),
            "selected_keys": selected_keys,
            "selection_strategy": "privacy-aware score = 0.5 update + 0.3 EMA history + 0.2 layer size",
            "ema_alpha": EMA_ALPHA,
            "update_weight": UPDATE_WEIGHT,
            "history_weight": HISTORY_WEIGHT,
            "layer_weight": LAYER_WEIGHT,
            "top_ranked_scores": ranked_scores[:20],
            "ckks_usage": "privacy-aware adaptive secure transmission and overhead profiling",
            "aggregation_note": "Multi-Krum aggregation is kept unchanged; CKKS protects selected tensors.",
            "ckks_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
            "ckks_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
            "ckks_global_scale": CKKS_GLOBAL_SCALE,
            "selected_layer_scope": "privacy-aware adaptive selected trainable tensors",
            "selected_tensor_count": len(metadata["selected_keys"]),
            "selected_total_values": metadata["total_values"],
            "avg_encryption_time_sec": avg_encryption_time,
            "total_encryption_time_sec": total_encryption_time,
            "encrypted_aggregation_time_sec": encrypted_aggregation_time,
            "decryption_time_sec": decryption_time,
            "crypto_time_sec": crypto_time,
            "crypto_overhead_percent": crypto_overhead_percent,
            "avg_plain_selected_size_bytes": avg_plain_selected_size,
            "avg_encrypted_selected_size_bytes": avg_encrypted_size,
            "avg_size_expansion_ratio": avg_size_expansion,
            "selective_plain_upload_mb": selective_plain_upload_mb,
            "selective_encrypted_upload_mb": selective_encrypted_upload_mb,
            "full_model_communication_cost_mb": full_model_comm_mb,
            "round_time_sec": round_time,
            "adaptive_encryption_metrics_per_client": adaptive_metric_rows,

            "avg_information_coverage_ratio": icr,
            "avg_residual_leakage_risk": rlr,
            "avg_reconstruction_risk_score": rrs,
            "avg_parameter_encryption_ratio": per,
            "avg_adaptive_encryption_quality": aeq,
            **adaptive_metrics,
            **metrics,
        }

        history.append(row)

        print(
            f"Round {round_idx} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Selected: {selected_indices} | "
            f"ICR: {icr:.4f} | "
            f"PER: {per:.6f} | "
            f"AEQ: {aeq:.2f} | "
            f"RLR/RRS: {rlr:.4f} | "
            f"Avg Enc: {avg_encryption_time:.4f}s | "
            f"HE Agg: {encrypted_aggregation_time:.4f}s | "
            f"Dec: {decryption_time:.4f}s | "
            f"Crypto Overhead: {crypto_overhead_percent:.4f}% | "
            f"Enc Upload: {selective_encrypted_upload_mb:.4f} MB | "
            f"Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    final_results = {
        "experiment": "EXP-007H",
        "method": "Privacy-Aware Adaptive CKKS with Multi-Krum Byzantine Defense",
        "dataset": "COVID Radiography Binary",
        "setting": "Moderate Non-IID + Byzantine sign-flip attack",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "model": "EfficientNet-B0",
        "aggregation": "Multi-Krum",
        "he_library": "TenSEAL",
        "he_scheme": "CKKS",
        "selection_strategy": "privacy-aware weighted score",
        "score_formula": "0.5*UpdateNorm + 0.3*EMAHistoricalImportance + 0.2*LayerImportance",
        "max_selected_bytes": MAX_SELECTED_BYTES,
        "ema_alpha": EMA_ALPHA,
        "update_weight": UPDATE_WEIGHT,
        "history_weight": HISTORY_WEIGHT,
        "layer_weight": LAYER_WEIGHT,
        "ckks_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
        "ckks_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
        "ckks_global_scale": CKKS_GLOBAL_SCALE,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
        "best_metrics": max(history, key=lambda row: row["accuracy"]),
    }

    with open(RESULTS_DIR / "covid_multikrum_privacy_aware_ckks.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(
        global_model.state_dict(),
        RESULTS_DIR / "covid_multikrum_privacy_aware_ckks.pth",
    )

    print("\nSaved Privacy-Aware Multi-Krum + CKKS results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()