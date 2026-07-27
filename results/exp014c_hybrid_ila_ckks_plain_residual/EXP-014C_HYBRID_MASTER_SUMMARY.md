# EXP-014C Hybrid ILA-CKKS + Plaintext Residual Summary

EXP-014C evaluates a practical selective-encryption workflow: ILA-selected high-leakage tensors are CKKS/HE aggregated, while non-selected lower-risk residual tensors are plaintext-aggregated from the same Multi-Krum-selected clients.

| Budget | HE-sensitive Keys | HE-sensitive PER | Plaintext Residual PER | Final Acc | Final F1 | Best Acc | Encrypted Upload | Crypto Overhead | Chain Valid | Tamper |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2,000,000 | 105 | 12.47% | 87.53% | 94.50% | 94.48% | 94.50% | 156.85 MB | 17.50% | True | True |
| 4,000,000 | 139 | 24.94% | 75.06% | 94.50% | 94.48% | 94.50% | 312.43 MB | 26.43% | True | True |
| 8,000,000 | 158 | 49.87% | 50.13% | 94.50% | 94.48% | 94.50% | 623.55 MB | 38.82% | True | True |

## Correct Interpretation

EXP-014C is not a full-HE experiment. It is a hybrid privacy-utility experiment. The sensitive selected tensors are protected through CKKS and HE aggregation; the residual tensors are intentionally exposed to preserve utility and reduce encryption cost.
