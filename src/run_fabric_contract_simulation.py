from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from blockchain.fabric_contract import FabricFLContract, sha256_hex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--output-dir", default="results/blockchain/fabric_contract_simulation")
    parser.add_argument(
        "--algorithm",
        default="EfficientNet-B0 + Multi-Krum + ILA CKKS + Fabric-style Blockchain",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = FabricFLContract()

    admin = {"role": "trusted_authority", "msp": "HealthcareTA"}
    edge = {"role": "edge_server", "msp": "EdgeOrg", "edge_server_id": "edge_server_1"}

    hospitals = [f"hospital_{i}" for i in range(1, args.num_clients + 1)]

    timings = {
        "register_hospital_ms": [],
        "submit_update_ms": [],
        "submit_aggregation_ms": [],
        "issue_rewards_ms": [],
        "query_round_ms": [],
    }

    for hospital in hospitals:
        start = time.perf_counter()
        contract.register_hospital(
            hospital_id=hospital,
            organization=f"HospitalOrg{i}" if (i := hospital.split("_")[-1]) else hospital,
            public_key_hash=sha256_hex({"hospital": hospital, "key": "public_key_placeholder"}),
            invoker=admin,
        )
        timings["register_hospital_ms"].append((time.perf_counter() - start) * 1000)

    for r in range(1, args.rounds + 1):
        selected_hospitals = hospitals[:]

        for hospital in hospitals:
            invoker = {"role": "hospital", "msp": f"{hospital}MSP", "hospital_id": hospital}

            update_hash = sha256_hex(
                {"round": r, "hospital": hospital, "type": "local_model_update", "algorithm": args.algorithm}
            )
            encrypted_update_hash = sha256_hex(
                {"round": r, "hospital": hospital, "plain_update_hash": update_hash, "encryption": "CKKS"}
            )

            start = time.perf_counter()
            contract.submit_update_hash(
                round_number=r,
                hospital_id=hospital,
                update_hash=update_hash,
                encrypted_update_hash=encrypted_update_hash,
                sample_count=400,
                local_metrics={"accuracy_proxy": round(0.80 + r * 0.03, 4)},
                invoker=invoker,
            )
            timings["submit_update_ms"].append((time.perf_counter() - start) * 1000)

        aggregation_hash = sha256_hex(
            {"round": r, "selected_hospitals": selected_hospitals, "operation": "Multi-Krum secure aggregation"}
        )
        global_model_hash = sha256_hex(
            {"round": r, "algorithm": args.algorithm, "model": "off_chain_global_model_pointer"}
        )

        start = time.perf_counter()
        contract.submit_aggregation(
            round_number=r,
            edge_server_id="edge_server_1",
            algorithm=args.algorithm,
            selected_hospitals=selected_hospitals,
            aggregation_hash=aggregation_hash,
            global_model_hash=global_model_hash,
            global_metrics={"accuracy": round(0.80 + r * 0.03, 4), "f1": round(0.80 + r * 0.03, 4)},
            invoker=edge,
        )
        timings["submit_aggregation_ms"].append((time.perf_counter() - start) * 1000)

        rewards = {hospital: round(1.0 / len(hospitals), 6) for hospital in hospitals}

        start = time.perf_counter()
        contract.issue_rewards(round_number=r, rewards=rewards, invoker=edge)
        timings["issue_rewards_ms"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        contract.query_round(r)
        timings["query_round_ms"].append((time.perf_counter() - start) * 1000)

    contract.export(output_dir)

    verification = contract.verify_contract_state()

    def avg(values):
        return round(sum(values) / len(values), 6) if values else 0.0

    metrics = {
        "experiment_id": "EXP-010",
        "experiment_name": "Hyperledger Fabric-style FL Smart Contract Abstraction",
        "rounds": args.rounds,
        "num_hospitals": args.num_clients,
        "algorithm": args.algorithm,
        "channel_name": verification["channel_name"],
        "world_state_assets": verification["world_state_assets"],
        "history_transactions": verification["history_transactions"],
        "world_state_root_hash": verification["world_state_root_hash"],
        "history_root_hash": verification["history_root_hash"],
        "avg_register_hospital_ms": avg(timings["register_hospital_ms"]),
        "avg_submit_update_ms": avg(timings["submit_update_ms"]),
        "avg_submit_aggregation_ms": avg(timings["submit_aggregation_ms"]),
        "avg_issue_rewards_ms": avg(timings["issue_rewards_ms"]),
        "avg_query_round_ms": avg(timings["query_round_ms"]),
        "fabric_style_functions": [
            "RegisterHospital",
            "SubmitUpdateHash",
            "SubmitAggregation",
            "IssueRewards",
            "QueryRound",
            "QueryHospitalHistory",
            "VerifyContractState",
        ],
        "access_control": {
            "trusted_authority": ["RegisterHospital"],
            "hospital": ["SubmitUpdateHash"],
            "edge_server": ["SubmitAggregation", "IssueRewards"],
        },
        "base_paper_extension": "Models the base paper's hospitals, edge servers, encrypted update transactions, aggregation records, and contribution rewards as Fabric-style smart contract assets.",
        "limitation": "This is a Fabric-style chaincode abstraction, not a deployed Hyperledger Fabric network with Docker peers/orderers.",
    }

    with (output_dir / "fabric_contract_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()