# Master Summary Insert — EXP-014C Hybrid ILA-CKKS + Plaintext Residual

EXP-014C completes the practical framework path by combining ILA-selected sensitive tensor encryption with plaintext residual aggregation. It resolves the utility issue observed in fully selective active-submodel training by allowing the full EfficientNet-B0 model to keep learning while still protecting the tensors identified as most leakage-sensitive by ILA.


## Key Result

| Budget | Sensitive HE Ratio | Plaintext Residual Ratio | Final Accuracy | Final F1 | Best Round | Final PCR | Residual Privacy Leakage | Crypto Overhead | Encrypted Upload | Final Round Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 12.47% | 87.53% | 94.50% | 94.48% | R4 | 99.9196% | 0.0804% | 17.50% | 156.85 MB | 60.60s |
| 4 MB | 24.94% | 75.06% | 94.50% | 94.48% | R4 | 99.9553% | 0.0447% | 26.43% | 312.43 MB | 71.42s |
| 8 MB | 49.87% | 50.13% | 94.50% | 94.48% | R4 | 99.9904% | 0.0096% | 38.82% | 623.55 MB | 113.49s |

## Final Interpretation

EXP-014C demonstrates that the framework can preserve strong model utility while applying CKKS only to ILA-selected sensitive tensors. The 2 MB hybrid setting is the recommended practical configuration because it reaches the same final utility as the larger budgets with far lower encrypted communication and crypto overhead. The 8 MB setting is the stronger privacy setting because it encrypts a larger fraction of the trainable model and further lowers residual privacy leakage.


## Recommended Paper Framing

Use EXP-014A as the full HE correctness reference, EXP-014B as the fully selective active-submodel feasibility result, and EXP-014C as the final practical hybrid framework mode.
