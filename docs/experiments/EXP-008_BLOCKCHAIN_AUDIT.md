# EXP-008 — Proof-of-Authority Blockchain Audit Layer

## Objective

Implement a blockchain-based audit layer for the modern federated learning healthcare framework.

The purpose of this experiment is to reproduce and extend the blockchain component of the 2025 base paper:

**Blockchain-based federated learning with homomorphic encryption for privacy-preserving healthcare data sharing**

The base paper uses blockchain to record model update transactions, improve transparency, support decentralized trust, and distribute rewards to participating hospitals. Our implementation follows the same core idea but adapts it to the modern PyTorch-based FL pipeline used in this repository.

---

## Base Paper Blockchain Design

The base paper proposes a cross-silo healthcare FL setup where hospitals train locally and upload encrypted local model updates to blockchain-based edge servers.

Main blockchain roles in the base paper:

1. Record hospital model-update transactions.
2. Maintain transparency and integrity through a distributed ledger.
3. Use edge servers as blockchain-based aggregation points.
4. Track hospital contribution.
5. Distribute rewards through contribution-based incentives.
6. Reduce reliance on a centralized FL server.

The paper implements the blockchain experiment using Ganache Truffle GUI in a local Ethereum-style environment.

---

## Our Design Decision

Instead of starting with Ganache/Truffle, this experiment implements a lightweight **Proof-of-Authority (PoA) audit ledger** in Python.

This was chosen because:

1. The main repository is Python/PyTorch-based.
2. A pure Python blockchain is easier to integrate into existing FL scripts.
3. It avoids setup friction from Solidity, Ganache, Truffle, Node.js, contract migration, and Web3 dependencies.
4. It produces deterministic JSON logs that can be committed directly to GitHub.
5. It is enough to demonstrate blockchain auditability, tamper detection, and contribution tracking for research purposes.

This does not claim to be a production blockchain network. It is a research-grade audit blockchain simulation.

---

## Implemented Components

| Component | Implemented |
|---|---|
| Genesis block | Yes |
| Hash-linked blocks | Yes |
| Previous hash validation | Yes |
| Merkle root per block | Yes |
| Authorized validator check | Yes |
| Client update transactions | Yes |
| Encrypted update hash logging | Yes |
| Aggregation transaction | Yes |
| Global model hash pointer | Yes |
| Reward transaction | Yes |
| Tamper detection test | Yes |
| Raw patient data on-chain | No |
| Raw model weights on-chain | No |

---

## What is Stored On-Chain

Each communication round creates one block containing:

1. Client update transactions.
2. Hash of each local update.
3. Hash of each encrypted update.
4. Sample count metadata.
5. Local metric metadata.
6. Aggregation transaction.
7. Selected client list.
8. Aggregation hash.
9. Global model hash.
10. Global round metrics.
11. Reward transaction.

Only hashes and metadata are stored.

No raw medical images, raw patient records, raw gradients, raw model weights, or decrypted updates are stored on-chain.

---

## Security Properties Evaluated

| Metric | Meaning |
|---|---|
| Chain validity | Whether all block hashes, Merkle roots, and previous hashes are correct |
| Tamper detection | Whether modifying a past transaction invalidates the chain |
| Block creation time | Runtime overhead for adding audit blocks |
| Verification time | Time required to verify the full ledger |
| Ledger size | Storage overhead of blockchain audit logging |
| Transaction count | Number of auditable FL events recorded |

---

## Final Multi-Krum ILA CKKS Blockchain Audit Result

- Source result: `results/multikrum_ila_ckks/covid_multikrum_ila_ckks.json`
- Blocks including genesis: 6
- Training round blocks: 5
- Transactions: 31
- Clients/Hospitals: 4
- Validators: 3
- Average block creation time: 0.10874 ms
- Max block creation time: 0.1379 ms
- Verification time: 0.5587 ms
- Ledger size: 28.37207 KB
- Chain valid: true
- Tamper detected: true

This confirms that the final Multi-Krum + ILA CKKS experiment can be audited using the Proof-of-Authority blockchain layer.

## Comparison with Base Paper

| Aspect | 2025 Base Paper | Our EXP-008 |
|---|---|---|
| Blockchain environment | Ganache + Truffle GUI | Pure Python PoA audit ledger |
| ML stack | TensorFlow/Keras CNN | PyTorch EfficientNet-B0 |
| FL setting | Hospitals + edge servers | Hospitals/clients + validator edge servers |
| Stored data | Transactions/contributions | Update hashes, encrypted update hashes, aggregation hash, global model hash, rewards |
| Consensus style | Local Ethereum auto-mining | Authorized rotating PoA validators |
| Tamper verification | Conceptual / blockchain property | Explicit validation and tamper test |
| Research extension | Blockchain + HE | Blockchain + CKKS + FedAvg/FedDyn + Multi-Krum + modern audit metrics |

---

## Novelty Over Base Paper

This experiment extends the original paper by making the blockchain layer more directly measurable and reproducible.

The base paper demonstrates blockchain-based FL with HE using a local Ethereum setup. Our implementation adds:

1. Explicit tamper detection result.
2. Per-round block creation overhead.
3. Chain verification overhead.
4. Ledger size measurement.
5. Global model hash pointer logging.
6. Compatibility with Byzantine-resilient Multi-Krum experiments.
7. Compatibility with selective/adaptive CKKS experiment logs.
8. Git-trackable JSON ledger outputs.

---

## Conclusion

EXP-008 completes the blockchain component of the modern FL-HE healthcare framework.

The blockchain layer does not improve classification accuracy directly. Its purpose is to provide auditability, integrity, transparency, and contribution tracking.

In the final system, the learning quality comes from EfficientNet-B0, FedAvg/FedDyn, Multi-Krum, and CKKS-based privacy-preserving aggregation. The blockchain layer strengthens trust by making each FL round verifiable and tamper-evident.