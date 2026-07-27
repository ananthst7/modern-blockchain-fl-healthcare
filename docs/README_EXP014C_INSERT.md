## EXP-014C — Hybrid ILA-CKKS Sensitive Tensor Aggregation + Plaintext Residual Multi-Krum

EXP-014C is the practical selective-encryption mode. ILA-selected sensitive tensors are protected with CKKS and HE aggregation, while non-selected lower-risk residual tensors are plaintext-aggregated from the same Multi-Krum-selected clients to preserve full-model learning.

| Budget | Sensitive HE Ratio | Residual Plaintext Ratio | Final Accuracy | Final F1 | Final PCR | Residual Privacy Leakage | Crypto Overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 12.47% | 87.53% | 94.50% | 94.48% | 99.9196% | 0.0804% | 17.50% |
| 4 MB | 24.94% | 75.06% | 94.50% | 94.48% | 99.9553% | 0.0447% | 26.43% |
| 8 MB | 49.87% | 50.13% | 94.50% | 94.48% | 99.9904% | 0.0096% | 38.82% |

Best practical setting: **2 MB**, because it reaches **94.50% accuracy / 94.48% F1** while encrypting only **12.47%** of trainable values and preserving **99.9196% measured ILA privacy coverage**.

Best privacy-coverage setting: **8 MB**, because it encrypts **49.87%** of trainable values and reduces final residual privacy leakage to approximately **0.0096%**.

Paper-safe claim: EXP-014C is a **hybrid selective privacy** workflow, not full-model end-to-end encryption. The selected sensitive tensors are HE-protected; the residual tensors remain plaintext by design.
