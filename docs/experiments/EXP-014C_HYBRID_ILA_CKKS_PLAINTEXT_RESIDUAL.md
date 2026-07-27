# EXP-014C — Hybrid ILA-CKKS Sensitive Tensor Aggregation + Plaintext Residual Multi-Krum

## 1. Purpose

EXP-014C evaluates the most practical selective-encryption design in the framework. Instead of freezing the non-selected tensors as in EXP-014B, EXP-014C updates the full EfficientNet-B0 model every round by splitting the trainable update into two paths:

```text
High-leakage / ILA-selected sensitive tensors → CKKS encryption → HE-assisted Multi-Krum → HE aggregation
Lower-risk residual tensors                  → plaintext aggregation from the same Multi-Krum-selected clients
Merged global state                           → HE-sensitive aggregate + plaintext residual aggregate
```

This is not a full-privacy design like EXP-014A. It is a selective protection design: the server role does not receive plaintext ILA-selected sensitive updates, while the non-selected residual tensors remain visible in plaintext as an explicit utility/privacy tradeoff.

The scientific purpose is to test whether ILA can protect nearly all measured leakage signal while preserving full-model learning quality and reducing HE cost compared with encrypting the whole trainable model.

## 2. Experimental Setup

| Item | Value |
|---|---|
| Experiment | EXP-014C |
| Dataset | COVID Radiography Binary |
| Model | EfficientNet-B0 |
| Setting | Moderate Non-IID + Byzantine sign-flip attack |
| Clients | 4 |
| Global rounds | 5 |
| Local epochs | 1 |
| Learning rate | 0.0001 |
| FedDyn alpha | 0.005 |
| Malicious client | Client index 0 |
| Attack scale | 5.0 |
| Robust aggregation | HE-assisted Multi-Krum over ILA-selected sensitive tensors |
| Residual aggregation | Plaintext weighted aggregation from the same Multi-Krum-selected clients |
| HE library/scheme | TenSEAL CKKS |
| Blockchain | PoA audit ledger |
| Seed | 7 |
| Device | CUDA |

## 3. Functional End-to-End Pipeline

```text
1. Each client receives the current global EfficientNet-B0 state.
2. Each client performs local FedDyn training on the full model.
3. The malicious client applies a sign-flip attack to its update.
4. Each client computes ILA metadata from update norm, Fisher-like score and gradient variance.
5. ILA selects high-leakage trainable tensors under the byte budget.
6. Selected sensitive tensors are CKKS-encrypted.
7. Pairwise distances are computed over selected encrypted tensors and decrypted as scalar scores.
8. Multi-Krum selects robust clients using those HE-derived selected-tensor distances.
9. Selected sensitive tensors from robust clients are HE-averaged.
10. Non-selected residual tensors from the same robust clients are plaintext-averaged.
11. The global model is updated by merging HE-sensitive tensors and plaintext residual tensors.
12. PoA blockchain records audit metadata, selected clients, hashes, metrics and rewards.
```

## 4. Budget Sweep Summary

| Budget | Sensitive HE Ratio | Plaintext Residual Ratio | Final Accuracy | Final F1 | Best Round | Final PCR | Residual Privacy Leakage | Crypto Overhead | Encrypted Upload | Final Round Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 12.47% | 87.53% | 94.50% | 94.48% | R4 | 99.9196% | 0.0804% | 17.50% | 156.85 MB | 60.60s |
| 4 MB | 24.94% | 75.06% | 94.50% | 94.48% | R4 | 99.9553% | 0.0447% | 26.43% | 312.43 MB | 71.42s |
| 8 MB | 49.87% | 50.13% | 94.50% | 94.48% | R4 | 99.9904% | 0.0096% | 38.82% | 623.55 MB | 113.49s |

## 5. Round-Wise Accuracy

| Budget | R1 | R2 | R3 | R4 | R5 |
|---:|---:|---:|---:|---:|---:|
| 2 MB | 81.00% | 89.50% | 94.00% | 94.50% | 94.50% |
| 4 MB | 81.00% | 89.25% | 94.25% | 94.50% | 94.50% |
| 8 MB | 81.00% | 89.25% | 94.00% | 94.50% | 94.50% |

## 6. Round-Wise F1 Score

| Budget | R1 | R2 | R3 | R4 | R5 |
|---:|---:|---:|---:|---:|---:|
| 2 MB | 80.47% | 89.41% | 93.98% | 94.48% | 94.48% |
| 4 MB | 80.47% | 89.16% | 94.23% | 94.48% | 94.48% |
| 8 MB | 80.47% | 89.16% | 93.98% | 94.48% | 94.48% |

## 7. Privacy and Leakage Coverage

| Budget | Final Sensitive HE Ratio | Final Residual Plaintext Ratio | Final PCR | Final RPL | Final LCR | Final VCR | Final InfCR | Gradient Energy Ratio Field |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 12.47% | 87.53% | 99.9196% | 0.0804% | 95.99% | 84.41% | 93.75% | 43.41% |
| 4 MB | 24.94% | 75.06% | 99.9553% | 0.0447% | 97.62% | 89.96% | 96.27% | 56.33% |
| 8 MB | 49.87% | 50.13% | 99.9904% | 0.0096% | 98.64% | 95.95% | 98.11% | 74.06% |

Interpretation: all three budgets preserve the same final utility, but larger budgets protect a larger fraction of trainable values and reduce measured residual privacy leakage. The 2 MB setting already reaches 99.9196% final PCR while encrypting only 12.47% of trainable values. The 8 MB setting reaches 99.9904% final PCR while encrypting 49.87% of trainable values.

## 8. Communication and Runtime

| Budget | Final HE Distance Time | Final HE Aggregation | Final Decryption | Final Encryption Total | Final Crypto Overhead | Final Encrypted Upload | Total Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 6.93s | 0.311s | 0.144s | 3.22s | 17.50% | 156.85 MB | 261.83s |
| 4 MB | 12.32s | 0.478s | 0.257s | 5.82s | 26.43% | 312.43 MB | 339.80s |
| 8 MB | 28.87s | 0.986s | 0.520s | 13.68s | 38.82% | 623.55 MB | 480.20s |

Runtime scales with encrypted sensitive budget. The 2 MB budget is the most efficient hybrid configuration, while 8 MB offers the strongest privacy coverage at higher overhead.

## 9. Robust Aggregation Behaviour

Across all budgets and all five rounds, HE-assisted Multi-Krum selected clients `[2, 3]`. The malicious sign-flip client was client index 0, and it was not selected in any round. The HE-derived distance matrices show the malicious client with distance scores several orders of magnitude larger than benign-client distances, making the Byzantine outlier clearly separable in the selected sensitive tensor space.

## 10. Blockchain Audit Summary

| Budget | Blocks incl. Genesis | Training Blocks | Transactions | Avg Block Time | Verification Time | Ledger Size | Chain Valid | Tamper Detected |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 MB | 6 | 5 | 31 | 0.2481 ms | 1.3766 ms | 28.57 KB | True | True |
| 4 MB | 6 | 5 | 31 | 0.2579 ms | 1.0877 ms | 28.57 KB | True | True |
| 8 MB | 6 | 5 | 31 | 0.2713 ms | 1.3381 ms | 28.57 KB | True | True |

The blockchain layer does not change model learning. Its role is auditability: it records hashes, selected-client metadata, aggregation metrics, round metrics and reward transactions without storing raw images or full model weights.

## 11. Comparison with EXP-014A and EXP-014B

| Experiment | Scope | Budget | Final Accuracy | Final F1 | Meaning |
|---|---|---:|---:|---:|---|
| EXP-014A | Full-trainable HE-assisted Multi-Krum | 20 MB | 94.50% | 94.48% | Strongest full-trainable HE correctness reference; high HE cost |
| EXP-014B | ILA-selected active submodel only | 2 MB | 76.50% | 75.13% | Fully selective active-submodel path, but limited capacity |
| EXP-014B | ILA-selected active submodel only | 4 MB | 88.75% | 88.61% | Better capacity, still selective |
| EXP-014B | ILA-selected active submodel only | 8 MB | 95.25% | 95.24% | Strongest active-submodel selective result |
| EXP-014C | Hybrid HE-sensitive + plaintext residual | 2 MB | 94.50% | 94.48% | Practical full-model utility with selective sensitive-tensor HE |
| EXP-014C | Hybrid HE-sensitive + plaintext residual | 4 MB | 94.50% | 94.48% | Practical full-model utility with selective sensitive-tensor HE |
| EXP-014C | Hybrid HE-sensitive + plaintext residual | 8 MB | 94.50% | 94.48% | Practical full-model utility with selective sensitive-tensor HE |

EXP-014C is the best practical system-design result. It keeps full-model utility even at 2 MB because the non-selected residual tensors are still updated. EXP-014B remains scientifically useful because it proves a stricter fully selective active-submodel path. EXP-014A remains the full HE reference.

## 12. Main Findings

- Hybrid aggregation preserved final accuracy at 94.50% for 2 MB, 4 MB and 8 MB budgets.
- The 2 MB hybrid budget is the best efficiency/utility point: it encrypts only 12.47% of trainable values while reaching 94.50% accuracy and 94.48% F1.
- The 8 MB hybrid budget is the strongest privacy-coverage point: it encrypts 49.87% of trainable values and lowers final residual privacy leakage to approximately 0.0096%.
- Multi-Krum consistently selected benign clients [2, 3] and excluded malicious client 0.
- The PoA audit chain remained valid and detected tampering in all budget runs.
- EXP-014C supports the final paper claim that selective ILA-CKKS can protect measured high-leakage tensors while plaintext residual aggregation preserves full-model utility.

## 13. Paper-Safe Claims

- ILA-selected sensitive tensors are functionally used in the model-update path.
- Selected sensitive tensors are CKKS-encrypted and HE-aggregated.
- Multi-Krum selection is performed using HE-derived distances over the selected sensitive tensor subspace.
- Non-selected residual tensors are plaintext-aggregated from the same robust clients to preserve full-model learning.
- The server does not receive plaintext selected sensitive updates in the role-separated prototype.
- The server does receive plaintext non-selected residual updates; this is an explicit privacy-utility tradeoff.
- The trusted authority decrypts pairwise distance scores and the final selected sensitive aggregate.
- PoA blockchain provides immutable audit logging of the FL round metadata and tamper detection.

## 14. Claims to Avoid

- Do not claim EXP-014C is full-model end-to-end encrypted aggregation.
- Do not claim the server never sees any plaintext model updates.
- Do not claim the non-selected residual tensors leak nothing.
- Do not claim CKKS eliminates the need for a trusted authority.
- Do not claim the blockchain improves accuracy; it provides auditability.
- Do not describe the gradient cosine field as true cosine similarity unless the code is changed; in the current logs it should be treated as a gradient-energy coverage field.

## 15. Final Conclusion

EXP-014C should be presented as the practical final framework mode. It combines privacy-aware selective HE with robust aggregation and blockchain auditability while keeping full-model utility. The 2 MB setting is the most efficient deployment configuration because it obtains 94.50% accuracy with only 12.47% sensitive-tensor HE coverage. The 8 MB setting is the stronger privacy configuration because it increases encrypted sensitive coverage to 49.87% and reduces residual privacy leakage further. Together with EXP-014A and EXP-014B, this establishes a complete privacy-cost-utility spectrum: full HE correctness, fully selective active-submodel feasibility, and practical hybrid deployment.
