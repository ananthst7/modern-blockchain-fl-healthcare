# EXP-009 — PoA vs PoW Blockchain Consensus Comparison

## Objective

Compare Proof-of-Authority and Proof-of-Work blockchain audit ledgers for federated healthcare learning.

The goal is to justify why Proof-of-Authority is more suitable than Proof-of-Work for cross-silo healthcare federated learning.

---

## Motivation

The base paper uses blockchain with federated learning and homomorphic encryption. It records hospital model updates as blockchain transactions and distributes contribution-based rewards.

However, the base paper uses a local Ganache/Truffle environment and does not explicitly evaluate consensus mechanism suitability.

This experiment fills that gap by comparing:

1. Proof-of-Authority audit ledger
2. Proof-of-Work audit ledger

---

## Why This Matters

Healthcare federated learning is usually cross-silo, not public and anonymous.

The participants are known institutions such as:

- hospitals
- clinics
- edge servers
- trusted authority
- consortium administrators

Therefore, Proof-of-Authority is naturally suitable because validators can be pre-approved healthcare or edge-server entities.

Proof-of-Work is less suitable because it adds unnecessary mining overhead and energy cost even though participants are already permissioned.

---

## Implementation

Both ledgers record the same FL audit information:

- hospital/client update hashes
- encrypted update hashes
- aggregation hash
- global model hash
- global metrics
- reward transactions
- Merkle root
- previous block hash
- tamper verification

The only difference is the consensus method.

---

## Metrics Evaluated

| Metric | Meaning |
|---|---|
| Average block creation time | Time required to create one audit block |
| Max block creation time | Worst-case block creation overhead |
| Verification time | Full-chain validation time |
| Transaction count | Number of logged FL audit events |
| Chain validity | Whether the chain remains valid |
| Tamper detection | Whether modification is detected |
| Slowdown ratio | PoW block time divided by PoA block time |

---

## Expected Interpretation

PoA should be faster because it only requires an authorized validator to sign/create the block.

PoW should be slower because it must repeatedly search for a nonce until the block hash satisfies the difficulty target.

Both can detect tampering, but PoW adds unnecessary computation for this healthcare FL setting.

---

## Final EXP-009 Results

| Metric | PoA | PoW |
|---|---:|---:|
| Blocks including genesis | 6 | 6 |
| Training round blocks | 5 | 5 |
| Transactions | 31 | 31 |
| Average block creation time | 0.09064 ms | 116.01598 ms |
| Max block creation time | 0.1048 ms | 260.7508 ms |
| Verification time | 0.4323 ms | 0.4595 ms |
| Chain valid | true | true |
| Tamper detected | true | true |

## Key Finding

PoW was approximately **1279.96× slower** than PoA for the same FL audit workload.

This confirms that PoA is the better consensus choice for cross-silo healthcare FL because hospitals and edge validators are known permissioned entities.

## Base Paper Comparison

| Aspect | Base Paper | EXP-009 |
|---|---|---|
| Blockchain tool | Ganache/Truffle | Python PoA and PoW audit ledgers |
| Consensus analysis | Not explicit | Explicit PoA vs PoW comparison |
| Participants | Hospitals and edge servers | Hospitals and authorized edge validators |
| Transaction logging | Yes | Yes |
| Contribution rewards | Yes | Yes |
| Tamper detection | Blockchain property | Explicitly tested |
| Novelty | Blockchain + HE + FL | Consensus-aware blockchain design for healthcare FL |

---

## Research Contribution

EXP-009 adds a consensus-level analysis missing from the base paper.

The key contribution is:

> For cross-silo healthcare federated learning, Proof-of-Authority is a better blockchain consensus choice than Proof-of-Work because the participants are known and permissioned, while PoW introduces unnecessary mining overhead without improving the FL model or healthcare privacy guarantees.

---

## Conclusion

EXP-009 strengthens the blockchain component of the project.

EXP-008 proved that the FL process can be audited with a tamper-evident PoA blockchain ledger.

EXP-009 shows why PoA is the preferred consensus mechanism for this healthcare FL setting.