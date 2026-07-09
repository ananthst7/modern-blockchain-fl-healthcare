import argparse
import json
import math
import time
from pathlib import Path

import torch

from models.efficientnet import build_efficientnet_b0
from federated.client import train_client
from federated.fedavg import fedavg
from federated.server import evaluate_global_model
from privacy.dp import add_dp_noise_to_update


CLIENTS_ROOT = Path("data/processed/covid_clients")
TEST_DIR = Path("data/processed/covid_binary/test")

NUM_CLIENTS = 4
DEFAULT_GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 2

DEFAULT_NOISE_STD = 1e-6
DEFAULT_CLIP_NORM = 100.0
DEFAULT_DELTA = 1e-5


def approx_update_level_epsilon(noise_std, clip_norm, delta=1e-5):
    if noise_std <= 0:
        return float("inf")

    sensitivity_replace_one = 2.0 * clip_norm
    sensitivity_add_remove = clip_norm
    factor = math.sqrt(2.0 * math.log(1.25 / delta))

    return {
        "replace_one_update": {
            "sensitivity": sensitivity_replace_one,
            "epsilon": sensitivity_replace_one * factor / noise_std,
        },
        "add_remove_update": {
            "sensitivity": sensitivity_add_remove,
            "epsilon": sensitivity_add_remove * factor / noise_std,
        },
    }


def privacy_grade(epsilon):
    if epsilon <= 1:
        return "strong"
    if epsilon <= 10:
        return "moderate"
    if epsilon <= 100:
        return "weak"
    if epsilon <= 1000:
        return "very weak"
    return "negligible / utility-focused"


def estimate_communication_cost_mb(state_dict, num_clients):
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    upload_mb = (total_bytes * num_clients) / (1024 ** 2)
    download_mb = (total_bytes * num_clients) / (1024 ** 2)

    return upload_mb + download_mb


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedAvg with classifier/update-level DP noise for COVID binary classification."
    )

    parser.add_argument("--global-rounds", type=int, default=DEFAULT_GLOBAL_ROUNDS)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--clip-norm", type=float, default=DEFAULT_CLIP_NORM)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--target-epsilon", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="results/fedavg_dp")

    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    epsilon_accounting = approx_update_level_epsilon(
        noise_std=args.noise_std,
        clip_norm=args.clip_norm,
        delta=args.delta,
    )

    replace_eps = epsilon_accounting["replace_one_update"]["epsilon"]
    add_remove_eps = epsilon_accounting["add_remove_update"]["epsilon"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Update-level DP config:")
    print(f"  clip_norm: {args.clip_norm}")
    print(f"  noise_std: {args.noise_std}")
    print(f"  delta: {args.delta}")
    print(f"  approx epsilon replace-one: {replace_eps:.6f}")
    print(f"  approx epsilon add/remove: {add_remove_eps:.6f}")

    global_model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)

    client_dirs = [CLIENTS_ROOT / f"client_{i}" for i in range(1, NUM_CLIENTS + 1)]

    history = []
    start_total = time.time()

    global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}

    for round_idx in range(1, args.global_rounds + 1):
        print(f"\n===== Global Round {round_idx}/{args.global_rounds} =====")
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

            state = {k: v.cpu() for k, v in state.items()}
            state = add_dp_noise_to_update(
                client_state=state,
                global_state=global_state_cpu,
                noise_std=args.noise_std,
                clip_norm=args.clip_norm,
            )

            client_states.append(state)
            client_sizes.append(size)
            client_losses.append(loss)

        avg_state = fedavg(client_states, client_sizes)
        global_model.load_state_dict(avg_state)
        global_model.to(device)

        global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}

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
        "experiment": "EXP-006A",
        "method": "FedAvg + update-level Differential Privacy",
        "dataset": "COVID Radiography Binary",
        "num_clients": NUM_CLIENTS,
        "global_rounds": args.global_rounds,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LR,
        "noise_std": args.noise_std,
        "clip_norm": args.clip_norm,
        "delta": args.delta,
        "target_epsilon": args.target_epsilon,
        "update_level_dp_epsilon_accounting": {
            "formal_status": "approximate Gaussian mechanism bound, not formal Opacus/RDP accounting",
            "replace_one_update": {
                **epsilon_accounting["replace_one_update"],
                "privacy_grade": privacy_grade(replace_eps),
            },
            "add_remove_update": {
                **epsilon_accounting["add_remove_update"],
                "privacy_grade": privacy_grade(add_remove_eps),
            },
        },
        "device": str(device),
        "total_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(results_dir / "covid_fedavg_efficientnet.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(global_model.state_dict(), results_dir / "covid_fedavg_efficientnet.pth")

    print("\nSaved FedAvg results to:", results_dir)


if __name__ == "__main__":
    main()