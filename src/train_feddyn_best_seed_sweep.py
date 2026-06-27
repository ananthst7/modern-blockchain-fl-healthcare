from pathlib import Path
import json
import time
import random
import numpy as np

import torch

from models.efficientnet import build_efficientnet_b0
from federated.feddyn_client import train_feddyn_client
from federated.feddyn import initialize_h_states, update_h_state, feddyn_aggregate
from federated.server import evaluate_global_model


CLIENTS_ROOT = Path("data/processed/covid_clients_extreme_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")
RESULTS_DIR = Path("results/feddyn_best_seed_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 11, 21, 42, 77]

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
ALPHA = 0.005
NUM_CLASSES = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)
    return upload_mb + download_mb


def run_one_seed(seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==============================")
    print(f"Running FedDyn seed={seed}")
    print(f"==============================")
    print("Device:", device)

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
    best_acc = 0.0
    best_f1 = 0.0
    best_round = 0
    best_state = None

    start_total = time.time()

    for round_idx in range(1, GLOBAL_ROUNDS + 1):
        print(f"\n===== FedDyn Seed {seed} Round {round_idx}/{GLOBAL_ROUNDS} =====")
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

            state = {k: v.cpu() for k, v in state.items()}

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

        comm_cost_mb = estimate_communication_cost_mb(
            global_model.state_dict(),
            NUM_CLIENTS,
        )

        round_time = time.time() - round_start

        row = {
            "seed": seed,
            "round": round_idx,
            "avg_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "client_sizes": client_sizes,
            "alpha": ALPHA,
            "communication_cost_mb": comm_cost_mb,
            "round_time_sec": round_time,
            **metrics,
        }

        history.append(row)

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_f1 = metrics["f1"]
            best_round = round_idx
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            }

        print(
            f"Round {round_idx} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Best Acc: {best_acc:.4f} @ R{best_round} | "
            f"Comm: {comm_cost_mb:.2f} MB | "
            f"Time: {round_time:.2f}s"
        )

    total_time = time.time() - start_total

    seed_result = {
        "seed": seed,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "best_round": best_round,
        "final_accuracy": history[-1]["accuracy"],
        "final_f1": history[-1]["f1"],
        "total_time_sec": total_time,
        "history": history,
    }

    with open(RESULTS_DIR / f"seed_{seed}_results.json", "w") as f:
        json.dump(seed_result, f, indent=4)

    torch.save(
        best_state,
        RESULTS_DIR / f"seed_{seed}_best_model.pth",
    )

    return seed_result


def main():
    all_results = []

    for seed in SEEDS:
        result = run_one_seed(seed)
        all_results.append(result)

    best = max(all_results, key=lambda x: x["best_accuracy"])

    summary = {
        "experiment": "FedDyn stability seed sweep",
        "method": "FedDyn on Extreme Non-IID",
        "seeds": SEEDS,
        "alpha": ALPHA,
        "global_rounds": GLOBAL_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "all_results": all_results,
        "best_seed": best["seed"],
        "best_accuracy": best["best_accuracy"],
        "best_f1": best["best_f1"],
        "best_round": best["best_round"],
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n===== FedDyn Seed Sweep Summary =====")
    for r in all_results:
        print(
            f"seed={r['seed']} | "
            f"best_acc={r['best_accuracy']:.4f} | "
            f"best_f1={r['best_f1']:.4f} | "
            f"best_round={r['best_round']} | "
            f"final_acc={r['final_accuracy']:.4f}"
        )

    print("\nBest seed:", best["seed"])
    print("Best accuracy:", best["best_accuracy"])
    print("Best F1:", best["best_f1"])
    print("Best round:", best["best_round"])


if __name__ == "__main__":
    main()