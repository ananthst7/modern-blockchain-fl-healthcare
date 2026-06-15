from pathlib import Path
import json
import time

import torch

from models.efficientnet import build_efficientnet_b0
from privacy.opacus_client import train_opacus_client
from federated.fedavg import fedavg
from federated.server import evaluate_global_model


CLIENTS_ROOT = Path("data/processed/covid_clients")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/fedavg_opacus_dp")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 2

NOISE_MULTIPLIER = 0.2
MAX_GRAD_NORM = 0.5
DELTA = 1e-5


def estimate_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0

    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)

    return upload_mb + download_mb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    global_model = build_efficientnet_b0(
        num_classes=NUM_CLASSES,
        pretrained=True,
    ).to(device)

    client_dirs = [
        CLIENTS_ROOT / f"client_{i}"
        for i in range(1, NUM_CLIENTS + 1)
    ]

    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== Opacus DP FedAvg Global Round {round_idx}/{GLOBAL_ROUNDS} =====")
        round_start = time.time()

        client_states = []
        client_sizes = []
        client_losses = []
        client_epsilons = []

        for client_dir in client_dirs:
            state, size, loss, epsilon = train_opacus_client(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                local_epochs=LOCAL_EPOCHS,
                lr=LR,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                delta=DELTA,
            )

            state = {k: v.cpu() for k, v in state.items()}

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)
            client_epsilons.append(epsilon)

            print(
                f"{client_dir.name} | "
                f"loss={loss:.4f} | "
                f"epsilon={epsilon:.4f}"
            )

        avg_state = fedavg(client_states, client_sizes)

        global_model.load_state_dict(avg_state)
        global_model.to(device)

        metrics = evaluate_global_model(global_model, TEST_DIR, device)

        comm_cost_mb = estimate_communication_cost_mb(
            global_model.state_dict(),
            NUM_CLIENTS,
        )

        round_time = time.time() - round_start

        row = {
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "client_epsilons": client_epsilons,
            "avg_epsilon": sum(client_epsilons) / len(client_epsilons),
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
            "delta": DELTA,
            "communication_cost_mb": comm_cost_mb,
            "round_time_sec": round_time,
            **metrics,
        }

        history.append(row)

        print(
            f"Round {round_idx} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Avg Epsilon: {row['avg_epsilon']:.4f} | "
            f"Comm: {comm_cost_mb:.2f} MB | "
            f"Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    final_results = {
        "experiment": "EXP-006B",
        "method": "FedAvg + Opacus DP-SGD",
        "dataset": "COVID Radiography Binary",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "noise_multiplier": NOISE_MULTIPLIER,
        "max_grad_norm": MAX_GRAD_NORM,
        "delta": DELTA,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / "covid_fedavg_opacus_dp.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(
        global_model.state_dict(),
        RESULTS_DIR / "covid_fedavg_opacus_dp.pth",
    )

    print("\nSaved Opacus DP FedAvg results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()