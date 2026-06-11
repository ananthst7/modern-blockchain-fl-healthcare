from pathlib import Path
import json
import time

import torch

from models.efficientnet import build_efficientnet_b0
from federated.feddyn_client import train_feddyn_client
from federated.feddyn import initialize_h_states, update_h_state, feddyn_aggregate
from federated.server import evaluate_global_model


CLIENTS_ROOT = Path("data/processed/covid_clients_extreme_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/feddyn_extreme_noniid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
ALPHA = 0.01
NUM_CLASSES = 2


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

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)
    client_dirs = [CLIENTS_ROOT / f"client_{i}" for i in range(1, NUM_CLIENTS + 1)]

    h_states = initialize_h_states(global_model.state_dict(), NUM_CLIENTS)

    history = []
    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== FedDyn Global Round {round_idx}/{GLOBAL_ROUNDS} =====")
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
                alpha=ALPHA,
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
                alpha=ALPHA,
            )

        new_global_state = feddyn_aggregate(
            client_states=client_states,
            client_sizes=client_sizes,
            h_states=h_states,
            alpha=ALPHA,
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
            "alpha": ALPHA,
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
        "experiment": "EXP-004",
        "method": "FedDyn + EfficientNet-B0 + Extreme Non-IID clients",
        "dataset": "COVID Radiography Binary",
        "num_clients": NUM_CLIENTS,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "alpha": ALPHA,
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / "covid_feddyn_extreme_noniid.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(global_model.state_dict(), RESULTS_DIR / "covid_feddyn_extreme_noniid.pth")

    print("\nSaved FedDyn results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()