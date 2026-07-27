# EXP-014A — Full-trainable HE-assisted Multi-Krum with Functional ILA-CKKS Aggregation

## 1. Why EXP-014A was added

Earlier CKKS/ILA experiments correctly measured selective encryption behaviour, CKKS overhead, privacy coverage, encrypted upload size, and ILA ranking quality. However, those experiments did not make the encrypted aggregate the sole source of the global model update. EXP-014A was added to close that gap.

The main goal of EXP-014A is to verify that the final healthcare FL framework can run with CKKS in the actual model-update path:

```text
FedDyn local training
→ Byzantine sign-flip attack
→ ILA metadata ranking
→ CKKS encryption of selected trainable tensors
→ HE-derived pairwise distance scores
→ Multi-Krum robust client selection
→ CKKS weighted averaging of selected robust clients
→ trusted-authority decryption of aggregate
→ decrypted HE aggregate inserted into global model
→ PoA blockchain audit logging
```

This experiment is therefore the first EXP-014 run where the HE result is functionally used for training instead of being treated only as a sidecar privacy/overhead benchmark.

## 2. Experiment identity

| Field | Value |
|---|---|
| Experiment ID | EXP-014A |
| Source result label | EXP-014 |
| Method | HE-Assisted Multi-Krum with Functional ILA-CKKS Aggregation |
| Dataset | COVID Radiography Binary |
| Model | EfficientNet-B0 |
| FL optimizer | FedDyn |
| Aggregation | HE-assisted Multi-Krum |
| HE library | TenSEAL |
| HE scheme | CKKS |
| Blockchain | Proof-of-Authority audit ledger |
| Clients | 4 |
| Global rounds | 5 |
| Local epochs | 1 |
| Learning rate | 0.0001 |
| FedDyn alpha | 0.005 |
| Malicious client index | 0 |
| Attack | Byzantine sign-flip |
| Attack scale | 5.0 |
| Seed | 7 |
| Device | cuda |
| Total runtime | 675.62 s |

## 3. Why the 20 MB budget was used

The original 2 MB functional HE update was not enough because it updated only about 12.47% of trainable EfficientNet parameters. That made the global model learn from a partial update while most trainable tensors stayed unchanged. The model therefore collapsed close to random prediction.

EXP-014A uses:

```text
max_selected_bytes = 20,000,000
```

This selected all trainable EfficientNet-B0 tensors:

| Metric | Value |
|---|---:|
| Selected trainable keys | 213 |
| Selected plaintext bytes | 16,040,440 |
| Selected trainable values | 4,010,110 |
| Parameter encryption ratio | 100.00% |
| Privacy coverage ratio | 100.00% |
| Residual privacy leakage | 0.00% |

This means EXP-014A should be described as a **full-trainable HE-assisted Multi-Krum run**, not as a budget-compressed ILA run.

## 4. Final result

| Metric | Value |
|---|---:|
| Final accuracy | **94.50%** |
| Final precision | **95.05%** |
| Final recall | **94.50%** |
| Final F1 score | **94.48%** |
| Best accuracy | **94.50%** |
| Best F1 score | **94.48%** |
| Final loss | 0.155289 |
| Final confusion matrix | [[178, 22], [0, 200]] |

Final confusion matrix interpretation:

```text
Classes: ['COVID', 'Normal']
[[COVID predicted as COVID, COVID predicted as Normal],
 [Normal predicted as COVID, Normal predicted as Normal]]

[[178, 22], [0, 200]]
```

The final model correctly classified all 200 Normal test samples and misclassified 22 out of 200 COVID samples as Normal.

## 5. Round-wise performance

| Round | Accuracy | Precision | Recall | F1 | Selected Clients | Selected Keys | PER | PCR | Crypto Overhead | Round Time |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 81.00% | 84.79% | 81.00% | 80.47% | [2, 3] | 213 | 100.00% | 100.00% | 56.07% | 135.16s |
| 2 | 89.25% | 90.64% | 89.25% | 89.16% | [2, 3] | 213 | 100.00% | 100.00% | 56.34% | 135.01s |
| 3 | 94.00% | 94.44% | 94.00% | 93.98% | [2, 3] | 213 | 100.00% | 100.00% | 56.22% | 135.62s |
| 4 | 94.50% | 95.05% | 94.50% | 94.48% | [2, 3] | 213 | 100.00% | 100.00% | 56.60% | 136.62s |
| 5 | 94.50% | 95.05% | 94.50% | 94.48% | [2, 3] | 213 | 100.00% | 100.00% | 56.45% | 133.19s |

The model improved from 81.00% in Round 1 to 94.50% by Rounds 4 and 5. Multi-Krum selected clients `[2, 3]` in every round, consistently excluding the malicious sign-flip client.

## 6. HE and communication overhead

| Metric | Average / Final Value |
|---|---:|
| Average encryption time per client | 5.6450 s |
| Average HE distance time | 50.5757 s |
| Average encrypted aggregation time | 1.9021 s |
| Average decryption time | 1.0622 s |
| Average crypto overhead | 56.33% |
| Average plaintext selected upload | 61.19 MB |
| Average encrypted selected upload | 1249.67 MB |
| Average CKKS expansion ratio | 20.42x |
| Final full model communication cost baseline | 123.66 MB |

The major cost is HE distance computation, which averaged about 50.58 seconds per round. The encrypted upload is large because EXP-014A encrypts the full trainable model path, giving correctness at the cost of overhead.

## 7. Robust aggregation behaviour

Round 1 Krum scores:

```text
[[2.5665192423814593, 2], [2.5665192423814593, 3], [2.8838163069053917, 1], [4936243.266935623, 0]]
```

Final round Krum scores:

```text
[[2.307561680389563, 2], [2.307561680389563, 3], [3.1010387418095746, 1], [4974824.13077249, 0]]
```

The malicious client received an extremely large distance score compared with benign clients. Multi-Krum therefore selected `[2, 3]` in every round. This is important because it shows that the HE-assisted distance matrix preserved the robust-selection behaviour required by Multi-Krum.

## 8. Blockchain audit result

| Metric | Value |
|---|---:|
| Blockchain type | Proof-of-Authority audit ledger |
| Consensus | Authorized rotating validators |
| Validators | 3 |
| Blocks including genesis | 6 |
| Training round blocks | 5 |
| Transactions | 31 |
| Average block creation time | 0.263960 ms |
| Maximum block creation time | 0.280300 ms |
| Verification time | 1.202400 ms |
| Ledger size | 28.4375 KB |
| Chain valid | True |
| Tamper detected | True |

Blockchain data policy:

```text
Only hashes, metrics, selected-client metadata, distance-result metadata and reward records are stored on-chain.
```

## 9. Privacy architecture

EXP-014A uses a role-separated simulation:

```text
Client role:
  trains local model and encrypts selected trainable tensors

Aggregation-server role:
  receives encrypted selected tensors, metadata, hashes, sample counts, distance scores and selected indices

Trusted-authority role:
  decrypts pairwise distance scores and final selected-tensor aggregate

Blockchain role:
  stores hashes, selected-client metadata, metric metadata and rewards
```

The result records that the server role does not receive plaintext selected client updates and that the trusted authority decrypts distance scores and the final aggregate.

## 10. Correct claims

Use these claims in the paper:

1. EXP-014A demonstrates a functional HE-assisted robust aggregation path.
2. The encrypted aggregate is inserted into the global model update.
3. Multi-Krum selected benign clients `[2, 3]` and excluded the sign-flip client across all rounds.
4. The PoA ledger remained valid and detected tampering.
5. The cost of full-trainable CKKS is high, with roughly 56% crypto overhead and about 1.25 GB encrypted upload per round.

## 11. Claims to avoid

Do not claim:

1. EXP-014A is communication-efficient selective encryption.
2. EXP-014A proves that 2 MB ILA can train the whole model.
3. Multi-Krum is fully encrypted with no revealed scores.
4. There is no trusted-authority assumption.
5. The blockchain stores raw model updates or raw patient data.

## 12. Why EXP-014B is needed

EXP-014A proves correctness but uses full trainable encryption. That makes it strong for privacy correctness, but weak for ILA efficiency.

EXP-014B is therefore needed as the real selective/adaptive experiment:

```text
EXP-014B: ILA-selected active submodel HE-assisted Multi-Krum
Purpose: true selective/adaptive encryption with functional global update
Budgets: 2 MB / 4 MB / 8 MB
Rule: only selected tensors are trainable and HE-aggregated
```

The key change is conceptual:

```text
Wrong selective path:
  train full model → encrypt only 12% → update only 12% → expect full-model learning

Correct active-submodel path:
  select active tensors → train only those tensors → encrypt 100% of active tensors → HE-update the active submodel
```

This keeps ILA true to its name while preserving a functional end-to-end workflow.
