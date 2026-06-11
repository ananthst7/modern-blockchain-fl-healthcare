from pathlib import Path
import json
import time
import copy

import torch

from models.efficientnet import build_efficientnet_b0
from federated.feddyn_client import train_feddyn_client
from federated.feddyn import initialize_h_states, update_h_state, feddyn_aggregate
from federated.server import evaluate_global_model


CLIENTS_ROOT = Path("data/processed/covid_clients_extreme_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/feddyn_alpha_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 2

ALPHA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]


def estimate_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)

    return upload_mb + download_mb


def run_feddyn(alpha, device):
    print(f"\n==============================")
    print(f"Running FedDyn Alpha = {alpha}")
    print(f"==============================")

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    client_dirs = [CLIENTS_ROOT / f"client_{i}" for i in range(1, NUM_CLIENTS + 1)]
    h_states = initialize_h_states(global_model.state_dict(), NUM_CLIENTS)

    history = []
    start_total = time.time()

    best_acc = 0.0
    best_f1 = 0.0
    best_round = 0
    best_state = None

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== FedDyn Round {round_idx}/{GLOBAL_ROUNDS} | alpha={alpha} =====")
        round_start = time.time()

        old_global_state = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
            if torch.is_floating_point(v)
        }

        client_states = []
        client_sizes = []
        client_losses = []

        for client_idx, client_dir in enumerate(client_dirs):
            state, size, loss = train_feddyn_client(
                global_model=global_model,
                client_dir=client_dir,
                device=device,
                h_state=h_states[client_idx],
                alpha=alpha,
                local_epochs=LOCAL_EPOCHS,
                lr=LR,
            )

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)

            h_states[client_idx] = update_h_state(
                h_state=h_states[client_idx],
                client_state=state,
                global_state=old_global_state,
                alpha=alpha,
            )

        new_global_state = feddyn_aggregate(
            client_states=client_states,
            client_sizes=client_sizes,
            h_states=h_states,
            alpha=alpha,
        )

        global_model.load_state_dict(new_global_state)
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
            "alpha": alpha,
            **metrics,
        }

        history.append(row)

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_f1 = metrics["f1"]
            best_round = round_idx
            best_state = copy.deepcopy(global_model.state_dict())

        print(
            f"Round {round_idx} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Best Acc: {best_acc:.4f} | "
            f"Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    result = {
        "experiment": "EXP-004A",
        "method": "FedDyn Alpha Sweep + EfficientNet-B0 + Extreme Non-IID",
        "dataset": "COVID Radiography Binary",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "alpha": alpha,
        "device": str(device),
        "total_time_sec": total_time,
        "best_round": best_round,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / f"alpha_{alpha}.json", "w") as f:
        json.dump(result, f, indent=4)

    torch.save(best_state, RESULTS_DIR / f"best_alpha_{alpha}.pth")

    return {
        "alpha": alpha,
        "best_round": best_round,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "final_accuracy": history[-1]["accuracy"],
        "final_f1": history[-1]["f1"],
        "total_time_sec": total_time,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    summary = []

    for alpha in ALPHA_VALUES:
        result = run_feddyn(alpha, device)
        summary.append(result)

    with open(RESULTS_DIR / "alpha_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n===== Alpha Sweep Summary =====")
    for row in summary:
        print(
            f"alpha={row['alpha']} | "
            f"best_acc={row['best_accuracy']:.4f} | "
            f"best_f1={row['best_f1']:.4f} | "
            f"best_round={row['best_round']} | "
            f"final_acc={row['final_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()