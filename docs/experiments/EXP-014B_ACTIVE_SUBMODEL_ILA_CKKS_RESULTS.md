# EXP-014B — ILA-Selected Active Submodel HE-Assisted Multi-Krum

## 1. Purpose

EXP-014B evaluates whether selective/adaptive ILA encryption can be made part of the functional end-to-end federated learning path without pretending that a small encrypted subset updates the entire EfficientNet model.

The key design choice is that the ILA-selected tensors define the active trainable submodel. Only those tensors are locally trained, encrypted with CKKS, used for HE-assisted Multi-Krum distance scoring, homomorphically averaged, decrypted by the trusted authority, and inserted back into the global model. Non-selected trainable tensors remain frozen.

This makes ILA functional instead of sidecar: the selected tensors are not merely profiled after training; they are the actual trainable and encrypted global-update path.

## 2. Experimental Setup

| Item | Value |
|---|---|
| Dataset | COVID Radiography Binary |
| Model | EfficientNet-B0 |
| FL optimizer | FedDyn |
| Robust aggregation | HE-assisted Multi-Krum over the active selected submodel |
| Encryption | TenSEAL CKKS |
| Attack | Byzantine sign-flip attack |
| Malicious client | Client index 0 |
| Clients | 4 |
| Global rounds | 5 |
| Local epochs | 1 |
| FedDyn alpha | 0.005 |
| Learning rate | 0.0001 |
| Blockchain | PoA audit ledger |
| Seed | 7 |
| Device | CUDA |

## 3. Functional Pipeline

```text
Bootstrap ILA probe
→ select active trainable tensors under byte budget
→ freeze non-selected trainable tensors
→ client-side FedDyn training only on active tensors
→ sign-flip attack applied to malicious active update
→ CKKS encryption of active tensors
→ HE-derived pairwise distance scores
→ Multi-Krum robust client selection
→ CKKS weighted average of selected active tensors
→ trusted-authority decryption of final active aggregate
→ active submodel inserted into global model
→ BatchNorm/floating buffers synchronized from selected clients
→ PoA blockchain logs hashes, selected clients, metrics and rewards
```

## 4. Budget Sweep Results

| Budget | Active Keys | Active Trainable Values | Active PER | Final Accuracy | Final F1 | Best Accuracy | Best F1 | Final Crypto Overhead | Final Round Time | Encrypted Upload | Chain Valid | Tamper Detected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 MB | 69 | 500,000 | 12.47% | 76.50% | 75.13% | 76.50% | 75.13% | 20.34% | 46.54s | 156.84 MB | True | True |
| 4 MB | 102 | 1,000,000 | 24.94% | 88.75% | 88.61% | 88.75% | 88.61% | 31.25% | 58.93s | 312.42 MB | True | True |
| 8 MB | 109 | 1,999,998 | 49.87% | 95.25% | 95.24% | 95.25% | 95.24% | 45.96% | 81.51s | 623.56 MB | True | True |

## 5. Round-Wise Accuracy and F1

| Budget | R1 Accuracy | R2 Accuracy | R3 Accuracy | R4 Accuracy | R5 Accuracy | R5 F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 55.50% | 56.00% | 64.00% | 69.75% | 76.50% | 75.13% |
| 4 MB | 61.50% | 64.75% | 77.00% | 84.25% | 88.75% | 88.61% |
| 8 MB | 68.00% | 79.25% | 89.50% | 93.00% | 95.25% | 95.24% |

## 6. Result Interpretation

EXP-014B shows a clear monotonic privacy-cost-utility trend. With a 2 MB active submodel, only 12.47% of trainable values are active and encrypted, giving 76.50% final accuracy. With a 4 MB active submodel, 24.94% of trainable values are active and encrypted, improving final accuracy to 88.75%. With an 8 MB active submodel, 49.87% of trainable values are active and encrypted, reaching 95.25% final accuracy and 95.24% final F1.

This proves that selective ILA encryption can be part of the real functional update path, but it also shows that very aggressive budgets reduce model capacity. The 8 MB setting is the strongest active-submodel configuration because it keeps the selective ILA design while approaching the utility of the full HE run.

## 7. Comparison with EXP-014A

| Experiment | Functional Scope | Budget | Active/Encrypted Trainable Values | Final Accuracy | Final F1 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| EXP-014A | Full trainable HE-assisted Multi-Krum | 20 MB | 4,010,110 / 4,010,110 | 94.50% | 94.48% | Strongest full-trainable privacy-correct reference; high cost |
| EXP-014B | ILA-selected active submodel | 2 MB | 500,000 / 4,010,110 | 76.50% | 75.13% | Selective functional ILA path |
| EXP-014B | ILA-selected active submodel | 4 MB | 1,000,000 / 4,010,110 | 88.75% | 88.61% | Selective functional ILA path |
| EXP-014B | ILA-selected active submodel | 8 MB | 1,999,998 / 4,010,110 | 95.25% | 95.24% | Selective functional ILA path |

Important: EXP-014A and EXP-014B are not identical claims. EXP-014A proves that full-trainable HE-assisted Multi-Krum can produce a functional global model update. EXP-014B proves that ILA can define a smaller active trainable submodel that is trained, encrypted, robustly selected, HE-aggregated, and globally updated end-to-end.

## 8. Blockchain Audit Summary

| Budget | Blocks incl. Genesis | Transactions | Verification Time | Ledger Size | Chain Valid | Tamper Detected |
|---:|---:|---:|---:|---:|---|---|
| 2 MB | 6 | 31 | 1.1015 ms | 28.46 KB | True | True |
| 4 MB | 6 | 31 | 1.2266 ms | 28.49 KB | True | True |
| 8 MB | 6 | 31 | 1.0931 ms | 28.48 KB | True | True |

## 9. Valid Claims

- ILA-selected tensors define the active trainable submodel.
- The active submodel is trained, CKKS-encrypted, selected through HE-assisted Multi-Krum, homomorphically averaged, and globally updated.
- The server role does not receive plaintext active selected client updates in the role-separated prototype.
- The trusted authority decrypts pairwise distance scores and the final active-submodel aggregate.
- The PoA blockchain records hashes, metrics, selected-client metadata, active-submodel distance metadata, and reward records.
- A larger active ILA budget improves utility while increasing HE communication and crypto overhead.

## 10. Claims to Avoid

- Do not claim that a 2 MB budget updates the entire EfficientNet model.
- Do not claim fully encrypted Multi-Krum with no revealed distance scores.
- Do not claim there is no trusted-authority assumption.
- Do not claim that active-submodel ILA proves all unselected tensors leak nothing.
- Do not describe EXP-014B as full-model encrypted aggregation; that is EXP-014A.

## 11. Paper-Safe Conclusion

EXP-014B demonstrates that selective/adaptive ILA-CKKS can be integrated into the functional FL update path when selection, trainability, encryption, robust aggregation, and global update are aligned. The 8 MB active-submodel setting is the strongest selective result, reaching 95.25% accuracy while encrypting 49.87% of trainable parameters. Lower budgets provide stronger communication reduction but reduce model capacity and classification utility.
