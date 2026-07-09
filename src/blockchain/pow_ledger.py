"""
Proof-of-Work audit ledger for comparison with PoA.

This is not intended as the final healthcare blockchain design.
It exists for EXP-009 to show why PoA is more suitable for permissioned
cross-silo healthcare FL than PoW.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from blockchain.poa_ledger import Block, PoALedger, Transaction, merkle_root


class PoWLedger(PoALedger):
    def __init__(
        self,
        difficulty: int = 3,
        chain_id: str = "modern-fl-healthcare-pow",
        genesis_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.difficulty = difficulty
        super().__init__(
            validators=["pow_miner"],
            chain_id=chain_id,
            genesis_metadata=genesis_metadata or {},
        )

    def add_block(
        self,
        transactions: List[Transaction],
        validator: str = "pow_miner",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Block:
        finalized_transactions = [tx.finalize().to_dict() for tx in transactions]
        tx_hashes = [tx["tx_hash"] for tx in finalized_transactions]

        metadata = metadata or {}
        metadata["consensus"] = "proof_of_work"
        metadata["difficulty"] = self.difficulty

        nonce = 0
        prefix = "0" * self.difficulty
        start = time.perf_counter()

        while True:
            metadata["nonce"] = nonce

            block = Block(
                index=len(self.blocks),
                timestamp=time.time(),
                validator=validator,
                previous_hash=self.blocks[-1].block_hash or self.blocks[-1].compute_hash(),
                transactions=finalized_transactions,
                metadata=metadata,
                merkle_root=merkle_root(tx_hashes),
            ).finalize()

            if block.block_hash.startswith(prefix):
                metadata["mining_time_ms"] = round((time.perf_counter() - start) * 1000, 6)
                metadata["nonce"] = nonce

                block = Block(
                    index=block.index,
                    timestamp=block.timestamp,
                    validator=block.validator,
                    previous_hash=block.previous_hash,
                    transactions=block.transactions,
                    metadata=metadata,
                    merkle_root=block.merkle_root,
                ).finalize()

                self.blocks.append(block)
                return block

            nonce += 1