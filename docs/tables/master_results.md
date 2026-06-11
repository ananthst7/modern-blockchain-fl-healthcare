# Master Results Table

## COVID Dataset

| Experiment | Model | FL Algorithm | Encryption | Blockchain | Accuracy | F1 | Comm Cost (MB) | Training Time | Comparison to 2025 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| 2025 Paper Baseline | CNN | FedAvg | Pyfhel/SEAL | Ganache | 97.25% | N/A | N/A | N/A | Baseline |
| EXP-001 | EfficientNet-B0 | Centralized | None | None | 98.00% | 98.00% | 0 | N/A | +0.75% Accuracy |
| EXP-002 | EfficientNet-B0 | FedAvg | None | None | 97.50% | 97.50% | 123.66 | ~239.7 s | +0.25% Accuracy |
| EXP-003 | EfficientNet-B0 | FedAvg (Non-IID) | None | None | | | | | |
| EXP-004 | EfficientNet-B0 | FedDyn | None | None | | | | | |
| EXP-005 | EfficientNet-B0 | FedDyn + Multi-Krum | None | None | | | | | |
| EXP-006 | EfficientNet-B0 | FedDyn + Multi-Krum + DP | DP | None | | | | | |
| EXP-007 | EfficientNet-B0 | FedDyn + Multi-Krum + CKKS | CKKS | None | | | | | |
| EXP-008 | Full System | FedDyn + Multi-Krum + CKKS | CKKS | PoA | | | | | |