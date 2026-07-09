"""
Proof-of-Authority blockchain audit ledger for federated learning experiments.

Purpose:
- Record FL round events as blockchain transactions.
- Store only hashes and metadata, never raw medical data or raw model weights.
- Verify tamper resistance through chained block hashes and Merkle roots.
- Support contribution/reward tracking for hospitals/clients.

This is intentionally lightweight and reproducible for research experiments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: Any) -> str:
    if isinstance(data, (dict, list)):
        data = canonical_json(data).encode("utf-8")
    elif isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, bytes):
        pass
    else:
        data = str(data).encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def merkle_root(transaction_hashes: List[str]) -> str:
    if not transaction_hashes:
        return sha256_hex("EMPTY_MERKLE_ROOT")

    layer = transaction_hashes[:]

    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])

        next_layer = []
        for i in range(0, len(layer), 2):
            next_layer.append(sha256_hex(layer[i] + layer[i + 1]))

        layer = next_layer

    return layer[0]


@dataclass
class Transaction:
    tx_type: str
    round_number: int
    actor: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    tx_hash: Optional[str] = None

    def to_hash_payload(self) -> Dict[str, Any]:
        return {
            "tx_type": self.tx_type,
            "round_number": self.round_number,
            "actor": self.actor,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    def compute_hash(self) -> str:
        return sha256_hex(self.to_hash_payload())

    def finalize(self) -> "Transaction":
        self.tx_hash = self.compute_hash()
        return self

    def to_dict(self) -> Dict[str, Any]:
        if self.tx_hash is None:
            self.finalize()

        return {
            "tx_type": self.tx_type,
            "round_number": self.round_number,
            "actor": self.actor,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "tx_hash": self.tx_hash,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Transaction":
        tx = Transaction(
            tx_type=data["tx_type"],
            round_number=int(data["round_number"]),
            actor=data["actor"],
            payload=data["payload"],
            timestamp=float(data["timestamp"]),
            tx_hash=data.get("tx_hash"),
        )
        return tx


@dataclass
class Block:
    index: int
    timestamp: float
    validator: str
    previous_hash: str
    transactions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    merkle_root: str
    block_hash: Optional[str] = None

    def to_hash_payload(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "validator": self.validator,
            "previous_hash": self.previous_hash,
            "transactions": self.transactions,
            "metadata": self.metadata,
            "merkle_root": self.merkle_root,
        }

    def compute_hash(self) -> str:
        return sha256_hex(self.to_hash_payload())

    def finalize(self) -> "Block":
        self.block_hash = self.compute_hash()
        return self

    def to_dict(self) -> Dict[str, Any]:
        if self.block_hash is None:
            self.finalize()

        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "validator": self.validator,
            "previous_hash": self.previous_hash,
            "transactions": self.transactions,
            "metadata": self.metadata,
            "merkle_root": self.merkle_root,
            "block_hash": self.block_hash,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Block":
        return Block(
            index=int(data["index"]),
            timestamp=float(data["timestamp"]),
            validator=data["validator"],
            previous_hash=data["previous_hash"],
            transactions=data["transactions"],
            metadata=data["metadata"],
            merkle_root=data["merkle_root"],
            block_hash=data.get("block_hash"),
        )


class PoALedger:
    def __init__(
        self,
        validators: List[str],
        chain_id: str = "modern-fl-healthcare-poa",
        genesis_metadata: Optional[Dict[str, Any]] = None,
    ):
        if not validators:
            raise ValueError("At least one validator is required for PoA consensus.")

        self.validators = validators
        self.chain_id = chain_id
        self.blocks: List[Block] = []

        self._create_genesis_block(genesis_metadata or {})

    def _create_genesis_block(self, genesis_metadata: Dict[str, Any]) -> None:
        tx = Transaction(
            tx_type="GENESIS",
            round_number=0,
            actor="trusted_authority",
            payload={
                "chain_id": self.chain_id,
                "validators": self.validators,
                "note": "Genesis block for FL healthcare blockchain audit ledger.",
            },
        ).finalize()

        block = Block(
            index=0,
            timestamp=time.time(),
            validator=self.validators[0],
            previous_hash="0" * 64,
            transactions=[tx.to_dict()],
            metadata=genesis_metadata,
            merkle_root=merkle_root([tx.tx_hash]),
        ).finalize()

        self.blocks.append(block)

    def add_block(
        self,
        transactions: List[Transaction],
        validator: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Block:
        if validator not in self.validators:
            raise ValueError(f"Validator '{validator}' is not authorized.")

        finalized_transactions = [tx.finalize().to_dict() for tx in transactions]
        tx_hashes = [tx["tx_hash"] for tx in finalized_transactions]

        block = Block(
            index=len(self.blocks),
            timestamp=time.time(),
            validator=validator,
            previous_hash=self.blocks[-1].block_hash or self.blocks[-1].compute_hash(),
            transactions=finalized_transactions,
            metadata=metadata or {},
            merkle_root=merkle_root(tx_hashes),
        ).finalize()

        self.blocks.append(block)
        return block

    def validate_chain(self) -> Dict[str, Any]:
        errors: List[str] = []

        for i, block in enumerate(self.blocks):
            block_dict = block.to_dict()

            if block.validator not in self.validators:
                errors.append(f"Block {i}: unauthorized validator {block.validator}")

            recomputed_tx_hashes = []
            for tx_dict in block.transactions:
                tx = Transaction.from_dict(tx_dict)
                recomputed_tx_hash = tx.compute_hash()
                recomputed_tx_hashes.append(recomputed_tx_hash)

                if tx_dict.get("tx_hash") != recomputed_tx_hash:
                    errors.append(f"Block {i}: transaction hash mismatch")

            expected_merkle = merkle_root(recomputed_tx_hashes)
            if block.merkle_root != expected_merkle:
                errors.append(f"Block {i}: Merkle root mismatch")

            expected_block_hash = block.compute_hash()
            if block.block_hash != expected_block_hash:
                errors.append(f"Block {i}: block hash mismatch")

            if i == 0:
                if block.previous_hash != "0" * 64:
                    errors.append("Genesis block has invalid previous_hash")
            else:
                prev_hash = self.blocks[i - 1].block_hash
                if block.previous_hash != prev_hash:
                    errors.append(f"Block {i}: previous_hash mismatch")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "num_blocks": len(self.blocks),
            "num_transactions": sum(len(block.transactions) for block in self.blocks),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "validators": self.validators,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "PoALedger":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ledger = object.__new__(PoALedger)
        ledger.chain_id = data["chain_id"]
        ledger.validators = data["validators"]
        ledger.blocks = [Block.from_dict(block) for block in data["blocks"]]
        return ledger


def make_client_update_transaction(
    round_number: int,
    client_id: str,
    update_hash: str,
    encrypted_update_hash: str,
    sample_count: int,
    local_metrics: Dict[str, Any],
    selected_by_aggregator: bool,
) -> Transaction:
    return Transaction(
        tx_type="CLIENT_UPDATE",
        round_number=round_number,
        actor=client_id,
        payload={
            "update_hash": update_hash,
            "encrypted_update_hash": encrypted_update_hash,
            "sample_count": sample_count,
            "local_metrics": local_metrics,
            "selected_by_aggregator": selected_by_aggregator,
            "privacy_note": "Only hashes and metrics are stored on-chain; raw data and raw weights are off-chain.",
        },
    )


def make_aggregation_transaction(
    round_number: int,
    edge_server_id: str,
    algorithm: str,
    selected_clients: List[str],
    aggregation_hash: str,
    global_model_hash: str,
    global_metrics: Dict[str, Any],
) -> Transaction:
    return Transaction(
        tx_type="AGGREGATION",
        round_number=round_number,
        actor=edge_server_id,
        payload={
            "algorithm": algorithm,
            "selected_clients": selected_clients,
            "aggregation_hash": aggregation_hash,
            "global_model_hash": global_model_hash,
            "global_metrics": global_metrics,
        },
    )


def make_reward_transaction(
    round_number: int,
    edge_server_id: str,
    rewards: Dict[str, float],
) -> Transaction:
    return Transaction(
        tx_type="REWARD",
        round_number=round_number,
        actor=edge_server_id,
        payload={
            "reward_units": rewards,
            "reward_policy": "normalized contribution score based on accepted participation, sample count, and local metric quality",
        },
    )


def compute_rewards(
    clients: List[str],
    sample_counts: Dict[str, int],
    local_accuracies: Dict[str, float],
    selected_clients: List[str],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for client in clients:
        if client not in selected_clients:
            scores[client] = 0.0
            continue

        samples = float(sample_counts.get(client, 1))
        acc = float(local_accuracies.get(client, 0.0))
        scores[client] = samples * max(acc, 0.0)

    total = sum(scores.values())

    if total <= 0:
        return {client: 0.0 for client in clients}

    return {client: round(score / total, 6) for client, score in scores.items()}


def tamper_copy(ledger: PoALedger) -> PoALedger:
    data = copy.deepcopy(ledger.to_dict())
    loaded = object.__new__(PoALedger)
    loaded.chain_id = data["chain_id"]
    loaded.validators = data["validators"]
    loaded.blocks = [Block.from_dict(block) for block in data["blocks"]]

    if len(loaded.blocks) > 1:
        loaded.blocks[1].transactions[0]["payload"]["sample_count"] = 999999

    return loaded