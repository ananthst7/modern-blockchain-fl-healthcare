"""
EXP-009: PoA vs PoW blockchain consensus comparison.

Compares Proof-of-Authority and Proof-of-Work audit ledgers for FL healthcare logging.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from blockchain.poa_ledger import (
    PoALedger,
    compute_rewards,
    make_aggregation_transaction,
    make_client_update_transaction,
    make_reward_transaction,
    sha256_hex,
    tamper_copy,
)
from blockchain.pow_ledger import PoWLedger


def make_round_transactions(round_number: int, clients: List[str], algorithm: str):
    selected_clients = clients[:]
    accuracy = 0.80 + (round_number * 0.03)
    f1 = accuracy

    sample_counts = {client: 400 for client in clients}
    local_accuracies = {
        client: round(accuracy - 0.01 + i * 0.002, 4)
        for i, client in enumerate(clients)
    }

    txs = []

    for client in clients:
        update_hash = sha256_hex(
            {
                "round": round_number,
                "client": client,
                "algorithm": algorithm,
                "kind": "local_update",
            }
        )

        encrypted_update_hash = sha256_hex(
            {
                "update_hash": update_hash,
                "encryption": "CKKS",
                "client": client,
                "round": round_number,
            }
        )

        txs.append(
            make_client_update_transaction(
                round_number=round_number,
                client_id=client,
                update_hash=update_hash,
                encrypted_update_hash=encrypted_update_hash,
                sample_count=sample_counts[client],
                local_metrics={"accuracy_proxy": local_accuracies[client]},
                selected_by_aggregator=True,
            )
        )

    aggregation_hash = sha256_hex(
        {
            "round": round_number,
            "algorithm": algorithm,
            "selected_clients": selected_clients,
        }
    )

    global_model_hash = sha256_hex(
        {
            "round": round_number,
            "algorithm": algorithm,
            "accuracy": accuracy,
            "f1": f1,
        }
    )

    txs.append(
        make_aggregation_transaction(
            round_number=round_number,
            edge_server_id="edge_server_cluster_1",
            algorithm=algorithm,
            selected_clients=selected_clients,
            aggregation_hash=aggregation_hash,
            global_model_hash=global_model_hash,
            global_metrics={"accuracy": accuracy, "f1": f1},
        )
    )

    rewards = compute_rewards(
        clients=clients,
        sample_counts=sample_counts,
        local_accuracies=local_accuracies,
        selected_clients=selected_clients,
    )

    txs.append(
        make_reward_transaction(
            round_number=round_number,
            edge_server_id="edge_server_cluster_1",
            rewards=rewards,
        )
    )

    return txs


def benchmark_ledger(ledger, consensus_name: str, rounds: int, clients: List[str], algorithm: str):
    validators = ["edge_validator_1", "edge_validator_2", "edge_validator_3"]
    block_times = []

    for r in range(1, rounds + 1):
        txs = make_round_transactions(r, clients, algorithm)

        start = time.perf_counter()

        if consensus_name == "PoA":
            validator = validators[(r - 1) % len(validators)]
            ledger.add_block(
                transactions=txs,
                validator=validator,
                metadata={
                    "round": r,
                    "consensus": "proof_of_authority",
                    "reason": "permissioned healthcare edge validator",
                },
            )
        else:
            ledger.add_block(
                transactions=txs,
                validator="pow_miner",
                metadata={
                    "round": r,
                    "reason": "public-style mining comparison only",
                },
            )

        block_times.append((time.perf_counter() - start) * 1000)

    verify_start = time.perf_counter()
    validation = ledger.validate_chain()
    verification_time_ms = (time.perf_counter() - verify_start) * 1000

    tamper_validation = tamper_copy(ledger).validate_chain()

    return {
        "consensus": consensus_name,
        "chain_valid": validation["valid"],
        "tamper_detected": not tamper_validation["valid"],
        "num_blocks_including_genesis": validation["num_blocks"],
        "num_training_round_blocks": validation["num_blocks"] - 1,
        "num_transactions": validation["num_transactions"],
        "avg_block_creation_time_ms": round(statistics.mean(block_times), 6),
        "max_block_creation_time_ms": round(max(block_times), 6),
        "verification_time_ms": round(verification_time_ms, 6),
        "block_creation_times_ms": [round(x, 6) for x in block_times],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--pow-difficulty", type=int, default=3)
    parser.add_argument("--output-dir", default="results/blockchain/consensus_comparison")
    parser.add_argument(
        "--algorithm",
        default="EfficientNet-B0 + Multi-Krum + ILA CKKS + Blockchain Audit",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clients = [f"hospital_{i}" for i in range(1, args.num_clients + 1)]

    poa = PoALedger(
        validators=["edge_validator_1", "edge_validator_2", "edge_validator_3"],
        genesis_metadata={
            "experiment_id": "EXP-009",
            "consensus": "PoA",
            "purpose": "Healthcare FL audit ledger consensus comparison",
        },
    )

    pow_ledger = PoWLedger(
        difficulty=args.pow_difficulty,
        genesis_metadata={
            "experiment_id": "EXP-009",
            "consensus": "PoW",
            "purpose": "Comparison baseline only",
        },
    )

    poa_metrics = benchmark_ledger(
        ledger=poa,
        consensus_name="PoA",
        rounds=args.rounds,
        clients=clients,
        algorithm=args.algorithm,
    )

    pow_metrics = benchmark_ledger(
        ledger=pow_ledger,
        consensus_name="PoW",
        rounds=args.rounds,
        clients=clients,
        algorithm=args.algorithm,
    )

    comparison = {
        "experiment_id": "EXP-009",
        "experiment_name": "PoA vs PoW Blockchain Consensus Comparison",
        "rounds": args.rounds,
        "num_clients": args.num_clients,
        "pow_difficulty": args.pow_difficulty,
        "base_paper_gap": "The base paper uses Ganache/Truffle and a consortium blockchain setting but does not explicitly compare consensus mechanisms.",
        "our_novelty": "This experiment evaluates consensus suitability and shows why PoA is more appropriate for permissioned healthcare FL than PoW.",
        "poa": poa_metrics,
        "pow": pow_metrics,
        "summary": {
            "poa_avg_block_time_ms": poa_metrics["avg_block_creation_time_ms"],
            "pow_avg_block_time_ms": pow_metrics["avg_block_creation_time_ms"],
            "pow_to_poa_slowdown": round(
                pow_metrics["avg_block_creation_time_ms"] / max(poa_metrics["avg_block_creation_time_ms"], 1e-9),
                4,
            ),
            "recommendation": "Use PoA for cross-silo healthcare FL because hospitals and edge servers are known permissioned participants.",
        },
    }

    poa.save(output_dir / "poa_ledger.json")
    pow_ledger.save(output_dir / "pow_ledger.json")

    with (output_dir / "consensus_comparison_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()