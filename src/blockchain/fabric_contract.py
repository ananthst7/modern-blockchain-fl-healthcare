from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def now() -> float:
    return time.time()


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: Any) -> str:
    if isinstance(data, bytes):
        raw = data
    else:
        raw = canonical(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Hospital:
    hospital_id: str
    organization: str
    role: str
    public_key_hash: str
    active: bool
    registered_at: float


@dataclass
class UpdateAsset:
    update_id: str
    round_number: int
    hospital_id: str
    update_hash: str
    encrypted_update_hash: str
    sample_count: int
    local_metrics: Dict[str, Any]
    accepted: bool
    timestamp: float


@dataclass
class AggregationAsset:
    aggregation_id: str
    round_number: int
    edge_server_id: str
    algorithm: str
    selected_hospitals: List[str]
    aggregation_hash: str
    global_model_hash: str
    global_metrics: Dict[str, Any]
    timestamp: float


@dataclass
class RewardAsset:
    reward_id: str
    round_number: int
    hospital_id: str
    reward_score: float
    policy: str
    timestamp: float


class FabricAccessError(Exception):
    pass


class FabricContractError(Exception):
    pass


class FabricFLContract:
    """
    Hyperledger Fabric-style smart contract abstraction.

    This is not a live Fabric network.
    It models Fabric chaincode behavior:
    - world state
    - identity-based access control
    - asset creation
    - asset querying
    - immutable transaction history
    - endorsement-style policy checks
    """

    def __init__(self, channel_name: str = "fl-healthcare-channel"):
        self.channel_name = channel_name
        self.world_state: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []

    def _require_role(self, invoker: Dict[str, str], allowed_roles: List[str]) -> None:
        role = invoker.get("role")
        if role not in allowed_roles:
            raise FabricAccessError(f"Invoker role '{role}' not allowed. Required: {allowed_roles}")

    def _put_state(self, key: str, value: Dict[str, Any], tx_type: str, invoker: Dict[str, str]) -> None:
        self.world_state[key] = value
        self.history.append(
            {
                "tx_id": sha256_hex({"key": key, "value": value, "time": now()}),
                "tx_type": tx_type,
                "key": key,
                "value_hash": sha256_hex(value),
                "invoker": invoker,
                "timestamp": now(),
            }
        )

    def register_hospital(
        self,
        hospital_id: str,
        organization: str,
        public_key_hash: str,
        invoker: Dict[str, str],
    ) -> Dict[str, Any]:
        self._require_role(invoker, ["trusted_authority", "admin"])

        key = f"HOSPITAL::{hospital_id}"
        if key in self.world_state:
            raise FabricContractError(f"Hospital already registered: {hospital_id}")

        asset = Hospital(
            hospital_id=hospital_id,
            organization=organization,
            role="hospital",
            public_key_hash=public_key_hash,
            active=True,
            registered_at=now(),
        )

        self._put_state(key, asdict(asset), "RegisterHospital", invoker)
        return asdict(asset)

    def submit_update_hash(
        self,
        round_number: int,
        hospital_id: str,
        update_hash: str,
        encrypted_update_hash: str,
        sample_count: int,
        local_metrics: Dict[str, Any],
        invoker: Dict[str, str],
    ) -> Dict[str, Any]:
        self._require_role(invoker, ["hospital"])

        if invoker.get("hospital_id") != hospital_id:
            raise FabricAccessError("Hospital can submit only its own update.")

        hospital_key = f"HOSPITAL::{hospital_id}"
        if hospital_key not in self.world_state or not self.world_state[hospital_key]["active"]:
            raise FabricContractError(f"Hospital not active/registered: {hospital_id}")

        update_id = f"ROUND::{round_number}::UPDATE::{hospital_id}"
        if update_id in self.world_state:
            raise FabricContractError(f"Update already submitted: {update_id}")

        asset = UpdateAsset(
            update_id=update_id,
            round_number=round_number,
            hospital_id=hospital_id,
            update_hash=update_hash,
            encrypted_update_hash=encrypted_update_hash,
            sample_count=sample_count,
            local_metrics=local_metrics,
            accepted=True,
            timestamp=now(),
        )

        self._put_state(update_id, asdict(asset), "SubmitUpdateHash", invoker)
        return asdict(asset)

    def submit_aggregation(
        self,
        round_number: int,
        edge_server_id: str,
        algorithm: str,
        selected_hospitals: List[str],
        aggregation_hash: str,
        global_model_hash: str,
        global_metrics: Dict[str, Any],
        invoker: Dict[str, str],
    ) -> Dict[str, Any]:
        self._require_role(invoker, ["edge_server", "validator"])

        aggregation_id = f"ROUND::{round_number}::AGGREGATION"

        for hospital_id in selected_hospitals:
            update_id = f"ROUND::{round_number}::UPDATE::{hospital_id}"
            if update_id not in self.world_state:
                raise FabricContractError(f"Missing update for selected hospital: {hospital_id}")

        asset = AggregationAsset(
            aggregation_id=aggregation_id,
            round_number=round_number,
            edge_server_id=edge_server_id,
            algorithm=algorithm,
            selected_hospitals=selected_hospitals,
            aggregation_hash=aggregation_hash,
            global_model_hash=global_model_hash,
            global_metrics=global_metrics,
            timestamp=now(),
        )

        self._put_state(aggregation_id, asdict(asset), "SubmitAggregation", invoker)
        return asdict(asset)

    def issue_rewards(
        self,
        round_number: int,
        rewards: Dict[str, float],
        invoker: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        self._require_role(invoker, ["edge_server", "validator"])

        issued = []

        for hospital_id, reward_score in rewards.items():
            reward_id = f"ROUND::{round_number}::REWARD::{hospital_id}"
            asset = RewardAsset(
                reward_id=reward_id,
                round_number=round_number,
                hospital_id=hospital_id,
                reward_score=float(reward_score),
                policy="normalized accepted contribution score",
                timestamp=now(),
            )
            self._put_state(reward_id, asdict(asset), "IssueReward", invoker)
            issued.append(asdict(asset))

        return issued

    def query_round(self, round_number: int) -> Dict[str, Any]:
        prefix = f"ROUND::{round_number}::"
        assets = {k: v for k, v in self.world_state.items() if k.startswith(prefix)}
        return {"round_number": round_number, "assets": assets}

    def query_hospital_history(self, hospital_id: str) -> Dict[str, Any]:
        matched = {
            k: v
            for k, v in self.world_state.items()
            if v.get("hospital_id") == hospital_id
        }
        return {"hospital_id": hospital_id, "assets": matched}

    def verify_contract_state(self) -> Dict[str, Any]:
        asset_hashes = [sha256_hex(v) for _, v in sorted(self.world_state.items())]
        history_hashes = [sha256_hex(tx) for tx in self.history]

        return {
            "valid": True,
            "channel_name": self.channel_name,
            "world_state_assets": len(self.world_state),
            "history_transactions": len(self.history),
            "world_state_root_hash": sha256_hex(asset_hashes),
            "history_root_hash": sha256_hex(history_hashes),
        }

    def export(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        with (output / "fabric_world_state.json").open("w", encoding="utf-8") as f:
            json.dump(self.world_state, f, indent=2)

        with (output / "fabric_transaction_history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        with (output / "fabric_state_verification.json").open("w", encoding="utf-8") as f:
            json.dump(self.verify_contract_state(), f, indent=2)