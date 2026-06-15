from pathlib import Path
import json
import time

import torch

from models.efficientnet import build_efficientnet_b0
from federated.client import train_client
from federated.fedavg import fedavg
from federated.server import evaluate_global_model
from aggregation.attacks import sign_flip_attack
from aggregation.multikrum import multikrum


CLIENTS_ROOT = Path("data/processed/covid_clients_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/multikrum_byzantine_moderate_noniid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 2

MALICIOUS_CLIENT_INDEX = 0
ATTACK_SCALE = 5.0


def estimate_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)

    return upload_mb + download_mb


def run_experiment(aggregation_method: str, device):
    print(f"\n==============================")
    print(f"Running: {aggregation_method}")
    print(f"==============================")

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    client_dirs = [CLIENTS_ROOT / f"client_{i}" for i in range(1, NUM_CLIENTS + 1)]

    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== Round {round_idx}/{GLOBAL_ROUNDS} | {aggregation_method} =====")
        round_start = time.time()

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

            state = {k: v.cpu() for k, v in state.items()}

            if client_idx == MALICIOUS_CLIENT_INDEX:
                print(f"Applying sign-flip attack to {client_dir.name}")
                state = sign_flip_attack(state, scale=ATTACK_SCALE)

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)

        selected_indices = None
        krum_scores = None

        if aggregation_method == "FedAvg_Byzantine":
            new_global_state = fedavg(client_states, client_sizes)

        elif aggregation_method == "MultiKrum_Byzantine":
            new_global_state, selected_indices, krum_scores = multikrum(
                client_states=client_states,
                client_sizes=client_sizes,
                num_malicious=1,
                num_selected=2,
            )
            print("Multi-Krum selected clients:", selected_indices)

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        global_model.load_state_dict(new_global_state)
        global_model.to(device)

        metrics = evaluate_global_model(global_model, TEST_DIR, device)

        comm_cost_mb = estimate_communication_cost_mb(global_model.state_dict(), NUM_CLIENTS)
        round_time = time.time() - round_start

        row = {
            "round": round_idx,
            "aggregation_method": aggregation_method,
            "malicious_client_index": MALICIOUS_CLIENT_INDEX,
            "attack_scale": ATTACK_SCALE,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "selected_indices": selected_indices,
            "krum_scores": krum_scores,
            "communication_cost_mb": comm_cost_mb,
            "round_time_sec": round_time,
            **metrics,
        }

        history.append(row)

        print(
            f"Round {round_idx} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Comm: {comm_cost_mb:.2f} MB | "
            f"Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    final_results = {
        "experiment": "EXP-005",
        "method": aggregation_method,
        "dataset": "COVID Radiography Binary",
        "setting": "Extreme Non-IID + 1 Byzantine sign-flip client",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "malicious_client_index": MALICIOUS_CLIENT_INDEX,
        "attack_scale": ATTACK_SCALE,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / f"{aggregation_method}.json", "w") as f:
        json.dump(final_results, f, indent=4)

    return final_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    fedavg_results = run_experiment("FedAvg_Byzantine", device)
    multikrum_results = run_experiment("MultiKrum_Byzantine", device)

    summary = {
        "experiment": "EXP-005",
        "fedavg_byzantine_final_accuracy": fedavg_results["final_metrics"]["accuracy"],
        "fedavg_byzantine_final_f1": fedavg_results["final_metrics"]["f1"],
        "multikrum_byzantine_final_accuracy": multikrum_results["final_metrics"]["accuracy"],
        "multikrum_byzantine_final_f1": multikrum_results["final_metrics"]["f1"],
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n===== EXP-005 Summary =====")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()