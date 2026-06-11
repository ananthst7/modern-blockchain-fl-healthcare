from pathlib import Path
import json
import time

import torch

from models.efficientnet import build_efficientnet_b0
from federated.client import train_client
from federated.fedavg import fedavg
from federated.server import evaluate_global_model


CLIENTS_ROOT = Path("data/processed/covid_clients_extreme_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/fedavg_extreme_noniid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 2


def estimate_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    # each client uploads one model update per round
    upload_mb = (total_bytes * num_clients) / (1024 ** 2)

    # server sends global model to each client
    download_mb = (total_bytes * num_clients) / (1024 ** 2)

    return upload_mb + download_mb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)

    client_dirs = [CLIENTS_ROOT / f"client_{i}" for i in range(1, NUM_CLIENTS + 1)]

    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== Global Round {round_idx}/{GLOBAL_ROUNDS} =====")
        round_start = time.time()

        client_states = []
        client_sizes = []
        client_losses = []

        for client_dir in client_dirs:
            state, size, loss = train_client(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                local_epochs=LOCAL_EPOCHS,
                lr=LR,
            )

            # move state to CPU before aggregation to reduce GPU memory pressure
            state = {k: v.cpu() for k, v in state.items()}

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)

        avg_state = fedavg(client_states, client_sizes)
        global_model.load_state_dict(avg_state)
        global_model.to(device)

        metrics = evaluate_global_model(global_model, TEST_DIR, device)

        comm_cost_mb = estimate_communication_cost_mb(global_model.state_dict(), NUM_CLIENTS)
        round_time = time.time() - round_start

        row = {
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
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
        "experiment": "EXP-003B",
        "method": "FedAvg + EfficientNet-B0 + Extreme Non-IID clients",
        "dataset": "COVID Radiography Binary",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / "covid_fedavg_efficientnet.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(global_model.state_dict(), RESULTS_DIR / "covid_fedavg_efficientnet.pth")

    print("\nSaved FedAvg results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()