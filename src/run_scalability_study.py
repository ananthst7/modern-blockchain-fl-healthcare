"""
EXP-013: Scalability study for the full framework.

Measures how the final framework scales with the number of logical clients.
If only four prepared client folders exist, the script reuses those folders as
logical hospitals for system-level scalability measurement.

Run from repo root:
    python src/run_scalability_study.py --client-counts 4 8
    python src/run_scalability_study.py --client-counts 4 8 12
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from run_full_framework_integrated import run_integrated_framework


RESULTS_DIR = Path("results/scalability")
CLIENTS_ROOT = Path("data/processed/covid_clients_noniid")
TEST_DIR = Path("data/processed/covid_binary/test")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_run(num_clients: int, result: Dict[str, Any]) -> Dict[str, Any]:
    history = result["history"]
    final = result["final_metrics"]
    blockchain = result.get("blockchain_metrics") or {}

    return {
        "clients": num_clients,
        "accuracy": final.get("accuracy"),
        "f1": final.get("f1"),
        "best_accuracy": result.get("best_metrics", {}).get("accuracy"),
        "best_f1": result.get("best_metrics", {}).get("f1"),
        "avg_round_time_sec": mean([row.get("round_time_sec", 0.0) for row in history]),
        "final_round_time_sec": final.get("round_time_sec"),
        "avg_encryption_time_sec": mean([row.get("avg_encryption_time_sec", 0.0) for row in history]),
        "avg_he_aggregation_time_sec": mean([row.get("encrypted_aggregation_time_sec", 0.0) for row in history]),
        "avg_decryption_time_sec": mean([row.get("decryption_time_sec", 0.0) for row in history]),
        "avg_crypto_overhead_percent": mean([row.get("crypto_overhead_percent", 0.0) for row in history]),
        "final_encrypted_upload_mb": final.get("selective_encrypted_upload_mb"),
        "avg_encrypted_upload_mb": mean([row.get("selective_encrypted_upload_mb", 0.0) for row in history]),
        "final_per": final.get("avg_parameter_encryption_ratio"),
        "final_icr": final.get("avg_information_coverage_ratio"),
        "final_pcr": final.get("avg_privacy_coverage_ratio"),
        "final_lcr": final.get("avg_leakage_coverage_ratio"),
        "final_vcr": final.get("avg_variance_coverage_ratio"),
        "final_infcr": final.get("avg_influence_coverage_ratio"),
        "blockchain_tx_count": blockchain.get("num_transactions"),
        "ledger_size_kb": blockchain.get("ledger_size_kb"),
        "avg_block_creation_time_ms": blockchain.get("avg_block_creation_time_ms"),
        "blockchain_verification_time_ms": blockchain.get("verification_time_ms"),
        "chain_valid": blockchain.get("chain_valid"),
        "tamper_detected": blockchain.get("tamper_detected"),
        "client_dirs_reused": result.get("client_dirs_reused_for_scalability"),
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP-013 scalability study.")
    parser.add_argument("--client-counts", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--global-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-selected-bytes", type=int, default=2_000_000)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    for num_clients in args.client_counts:
        print(f"\n========== EXP-013 Scalability: {num_clients} clients ==========")
        run_dir = output_dir / f"clients_{num_clients}"
        result = run_integrated_framework(
            output_dir=run_dir,
            clients_root=CLIENTS_ROOT,
            test_dir=TEST_DIR,
            num_clients=num_clients,
            global_rounds=args.global_rounds,
            local_epochs=args.local_epochs,
            max_selected_bytes=args.max_selected_bytes,
            seed=args.seed,
            enable_blockchain=True,
            save_model=not args.no_save_model,
        )
        summary = summarize_run(num_clients, result)
        summaries.append(summary)

    payload = {
        "experiment": "EXP-013",
        "experiment_name": "Scalability Study",
        "method": "Full framework: FedDyn + Multi-Krum + ILA-CKKS + PoA blockchain",
        "client_counts": args.client_counts,
        "global_rounds": args.global_rounds,
        "local_epochs": args.local_epochs,
        "note": "If the requested logical client count exceeds the number of prepared client folders, the existing shards are reused cyclically. This measures system scaling more than new data diversity.",
        "summary": summaries,
    }

    with (output_dir / "scalability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=4)

    write_csv(summaries, output_dir / "scalability_summary.csv")

    print("\n===== EXP-013 SCALABILITY SUMMARY =====")
    print(json.dumps(json_safe(summaries), indent=4))
    print("\nSaved scalability results to:", output_dir)


if __name__ == "__main__":
    main()
