from pathlib import Path
import json
import time

import torch

from models.efficientnet import build_efficientnet_b0
from federated.feddyn_client import train_feddyn_client
from federated.feddyn import initialize_h_states, update_h_state, feddyn_aggregate
from federated.server import evaluate_global_model
from encryption.tenseal_ckks import (
    create_ckks_context,
    encrypt_selected_state,
    encrypted_weighted_average,
    decrypt_selected_state,
)


CLIENTS_ROOT = Path("data/processed/covid_clients_extreme_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/feddyn_selective_ckks")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
ALPHA = 0.005
NUM_CLASSES = 2

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


def merge_classifier_into_global(global_state, classifier_state):
    merged_state = {
        key: value.detach().cpu().clone()
        for key, value in global_state.items()
    }

    for key, value in classifier_state.items():
        if "classifier" in key and torch.is_floating_point(value):
            merged_state[key] = value.detach().cpu().clone()

    return merged_state


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

    h_states = initialize_h_states(global_model.state_dict(), NUM_CLIENTS)

    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== FedDyn + Selective CKKS Round {round_idx}/{GLOBAL_ROUNDS} =====")
        round_start = time.time()

        old_global_state = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_floating_point(v)
        }

        client_states = []
        client_sizes = []
        client_losses = []

        encrypted_vectors = []
        encryption_times = []
        encrypted_sizes = []
        plain_selected_sizes = []
        size_expansion_ratios = []
        metadata = None

        for client_idx, client_dir in enumerate(client_dirs):
            state, size, loss = train_feddyn_client(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                h_state=h_states[client_idx],
                alpha=ALPHA,
                local_epochs=LOCAL_EPOCHS,
                lr=LR,
            )

            state = {
                key: value.cpu()
                for key, value in state.items()
            }

            encrypted_vector, metadata, enc_metrics = encrypt_selected_state(
                state_dict=state,
                context=ckks_context,
            )

            encrypted_vectors.append(encrypted_vector)
            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)

            encryption_times.append(enc_metrics["encryption_time_sec"])
            encrypted_sizes.append(enc_metrics["encrypted_size_bytes"])
            plain_selected_sizes.append(enc_metrics["plain_size_bytes"])
            size_expansion_ratios.append(enc_metrics["size_expansion_ratio"])

            h_states[client_idx] = update_h_state(
                h_state=h_states[client_idx],
                client_state=state,
                global_state=old_global_state,
                alpha=ALPHA,
            )

            print(
                f"{client_dir.name} | "
                f"loss={loss:.4f} | "
                f"enc_time={enc_metrics['encryption_time_sec']:.4f}s | "
                f"plain={enc_metrics['plain_size_bytes']} bytes | "
                f"encrypted={enc_metrics['encrypted_size_bytes']} bytes | "
                f"expansion={enc_metrics['size_expansion_ratio']:.2f}x"
            )

        encrypted_avg, encrypted_aggregation_time = encrypted_weighted_average(
            encrypted_vectors=encrypted_vectors,
            client_sizes=client_sizes,
        )

        global_state_cpu = {
            key: value.detach().cpu().clone()
            for key, value in global_model.state_dict().items()
        }

        decrypted_classifier_state, decryption_time = decrypt_selected_state(
            encrypted_vector=encrypted_avg,
            metadata=metadata,
            reference_state=global_state_cpu,
        )

        # FedDyn aggregate full model normally, then replace classifier with CKKS aggregated classifier.
        feddyn_state = feddyn_aggregate(
            client_states=client_states,
            client_sizes=client_sizes,
            h_states=h_states,
            alpha=ALPHA,
        )

        new_global_state = merge_classifier_into_global(
            global_state=feddyn_state,
            classifier_state=decrypted_classifier_state,
        )

        global_model.load_state_dict(new_global_state)
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

        row = {
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "alpha": ALPHA,
            "ckks_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
            "ckks_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
            "ckks_global_scale": CKKS_GLOBAL_SCALE,
            "selected_layer_scope": "classifier-only",
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
            **metrics,
        }

        history.append(row)

        print(
            f"Round {round_idx} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Avg Enc: {avg_encryption_time:.4f}s | "
            f"HE Agg: {encrypted_aggregation_time:.4f}s | "
            f"Dec: {decryption_time:.4f}s | "
            f"Crypto Overhead: {crypto_overhead_percent:.4f}% | "
            f"Enc Upload: {selective_encrypted_upload_mb:.4f} MB | "
            f"Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    final_results = {
        "experiment": "EXP-007B",
        "method": "FedDyn + Selective CKKS Encrypted Aggregation using TenSEAL",
        "dataset": "COVID Radiography Binary",
        "setting": "Extreme Non-IID",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "alpha": ALPHA,
        "model": "EfficientNet-B0",
        "aggregation": "FedDyn",
        "he_library": "TenSEAL",
        "he_scheme": "CKKS",
        "selected_layer_scope": "classifier-only",
        "ckks_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
        "ckks_coeff_mod_bit_sizes": CKKS_COEFF_MOD_BIT_SIZES,
        "ckks_global_scale": CKKS_GLOBAL_SCALE,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / "covid_feddyn_selective_ckks.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(
        global_model.state_dict(),
        RESULTS_DIR / "covid_feddyn_selective_ckks.pth",
    )

    print("\nSaved FedDyn + Selective CKKS results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()