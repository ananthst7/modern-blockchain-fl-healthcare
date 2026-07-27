# EXP-014B Active Submodel Budget Summary

EXP-014B evaluates whether selective/adaptive ILA-CKKS can remain functional when the selected tensors define the active trainable submodel.

| Budget | Active Keys | Active Bytes | Active PER | Final Acc | Final F1 | Best Acc | Encrypted Upload | Crypto Overhead | Chain Valid | Tamper |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2,000,000 | 69 | 2,000,000 | 12.47% | 76.50% | 75.13% | 76.50% | 156.84 MB | 20.34% | True | True |
| 4,000,000 | 102 | 4,000,000 | 24.94% | 88.75% | 88.61% | 88.75% | 312.42 MB | 31.25% | True | True |
| 8,000,000 | 109 | 7,999,992 | 49.87% | 95.25% | 95.24% | 95.25% | 623.56 MB | 45.96% | True | True |

## Interpretation

If lower budgets preserve useful accuracy, this supports the claim that ILA can be used as a functional selective-encryption mechanism. If accuracy drops at lower budgets, that is still a valid result: it quantifies the utility cost of shrinking the active encrypted submodel.