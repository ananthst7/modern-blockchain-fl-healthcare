## EXP-014B — ILA-Selected Active Submodel HE-Assisted Multi-Krum

EXP-014B evaluates a true selective/adaptive encryption workflow. ILA first selects an active trainable submodel; only this selected submodel is trained, CKKS-encrypted, robustly selected by HE-assisted Multi-Krum, homomorphically averaged, and inserted into the global model.

| Budget | Active PER | Final Accuracy | Final F1 | Active Keys | Encrypted Upload | Crypto Overhead |
|---:|---:|---:|---:|---:|---:|---:|
| 2 MB | 12.47% | 76.50% | 75.13% | 69 | 156.84 MB | 20.34% |
| 4 MB | 24.94% | 88.75% | 88.61% | 102 | 312.42 MB | 31.25% |
| 8 MB | 49.87% | 95.25% | 95.24% | 109 | 623.56 MB | 45.96% |

Best selective setting: **8 MB**, reaching **95.25% accuracy** and **95.24% F1** while using **49.87%** of trainable parameters in the active encrypted submodel.

Paper-safe wording: EXP-014B is end-to-end for the ILA-selected active submodel, not a claim that the full EfficientNet model is updated under a tiny budget.
