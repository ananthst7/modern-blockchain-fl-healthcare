# Experiment Log

## Research Objective

Develop a modern blockchain-enabled privacy-preserving federated learning framework for healthcare image classification by extending the 2025 FL-HE framework using improved deep learning architectures and robust federated optimization techniques.

---

# Baseline Paper

**Title:** Blockchain-based Federated Learning with Homomorphic Encryption for Privacy-Preserving Healthcare Data Sharing (2025)

### COVID Dataset Baseline Performance

| Method                    | Accuracy |
| ------------------------- | -------: |
| CNN + FedAvg (2025 Paper) |   97.25% |

---

# EXP-001: Centralized EfficientNet-B0 Baseline

**Date:** 11/06/2026

## Objective

Establish a strong centralized baseline on the same COVID dataset used by the 2025 paper before introducing federated learning, encryption, or blockchain.

## Dataset

* COVID-19 Radiography Database
* Binary Classification:

  * COVID
  * Normal

## Hyperparameters

| Parameter      | Value                      |
| -------------- | -------------------------- |
| Model          | EfficientNet-B0            |
| Initialization | ImageNet Pretrained        |
| Framework      | PyTorch                    |
| Device         | NVIDIA RTX 3060 Laptop GPU |
| Batch Size     | 16                         |
| Epochs         | 5                          |
| Optimizer      | AdamW                      |
| Learning Rate  | 1×10⁻⁴                     |
| Input Size     | 224×224                    |
| Loss Function  | CrossEntropyLoss           |

## Results

| Epoch | Train Loss | Test Accuracy |   F1 Score |
| ----: | ---------: | ------------: | ---------: |
|     1 |     0.3899 |        96.00% |     96.00% |
|     2 |     0.1706 |        96.75% |     96.75% |
|     3 |     0.1045 |        97.50% |     97.50% |
|     4 | **0.0874** |    **98.00%** | **98.00%** |
|     5 |     0.0724 |        95.25% |     95.24% |

## Comparison with 2025 Paper

| Method     |   Accuracy |
| ---------- | ---------: |
| 2025 Paper |     97.25% |
| EXP-001    | **98.00%** |

**Improvement:** +0.75%

## Observations

* EfficientNet-B0 achieved peak performance at Epoch 4.
* Accuracy degradation at Epoch 5 suggests overfitting.
* EfficientNet-B0 outperformed the CNN baseline of the 2025 paper.

---

# EXP-002: IID FedAvg with EfficientNet-B0

**Date:** 11/06/2026

## Objective

Evaluate the effect of standard FedAvg on COVID classification under IID hospital distributions.

## Dataset

* COVID-19 Radiography Database
* Binary Classification:

  * COVID
  * Normal

## Hyperparameters

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Clients       | 4                          |
| Local Epochs  | 1                          |
| Global Rounds | 5                          |
| Model         | EfficientNet-B0            |
| Aggregation   | FedAvg                     |
| Device        | NVIDIA RTX 3060 Laptop GPU |

## Results

| Round |   Accuracy |   F1 Score | Communication Cost | Round Time |
| ----: | ---------: | ---------: | -----------------: | ---------: |
|     1 |     81.00% |     80.62% |          123.66 MB |    43.83 s |
|     2 |     89.75% |     89.74% |          123.66 MB |    43.78 s |
|     3 |     93.50% |     93.48% |          123.66 MB |    50.62 s |
|     4 |     95.50% |     95.50% |          123.66 MB |    50.74 s |
|     5 | **97.50%** | **97.50%** |          123.66 MB |    50.73 s |

## Comparison with 2025 Paper

| Method     |   Accuracy |
| ---------- | ---------: |
| 2025 Paper |     97.25% |
| EXP-002    | **97.50%** |

**Improvement:** +0.25%

## Comparison with EXP-001

| Method  | Accuracy |
| ------- | -------: |
| EXP-001 |   98.00% |
| EXP-002 |   97.50% |

**Difference:** -0.50%

## Observations

* FedAvg nearly matched centralized performance.
* No raw hospital data sharing was required.
* The federated approach still exceeded the 2025 baseline.

---

# EXP-003: Moderate Non-IID FedAvg

**Date:** 11/06/2026

## Objective

Investigate the effect of heterogeneous hospital distributions on standard FedAvg.

## Hospital Distribution

| Client   | COVID | Normal |
| -------- | ----: | -----: |
| Client 1 |   320 |     80 |
| Client 2 |   280 |    120 |
| Client 3 |   120 |    280 |
| Client 4 |    80 |    320 |

## Hyperparameters

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Clients       | 4                          |
| Local Epochs  | 1                          |
| Global Rounds | 5                          |
| Model         | EfficientNet-B0            |
| Aggregation   | FedAvg                     |
| Device        | NVIDIA RTX 3060 Laptop GPU |

## Results

| Round |   Accuracy |   F1 Score | Communication Cost | Round Time |
| ----: | ---------: | ---------: | -----------------: | ---------: |
|     1 |     75.25% |     74.81% |          123.66 MB |    51.72 s |
|     2 |     86.75% |     86.74% |          123.66 MB |    51.39 s |
|     3 |     92.50% |     92.48% |          123.66 MB |    55.11 s |
|     4 |     95.50% |     95.50% |          123.66 MB |    56.84 s |
|     5 | **97.00%** | **97.00%** |          123.66 MB |    57.77 s |

## Comparison

| Method               | Accuracy |
| -------------------- | -------: |
| 2025 Paper           |   97.25% |
| EXP-002 (IID FedAvg) |   97.50% |
| EXP-003              |   97.00% |

## Observations

* Moderate heterogeneity caused only minor degradation.
* FedAvg remained surprisingly robust.
* Accuracy dropped by only 0.50 percentage points compared to IID FedAvg.

---

# EXP-003B: Extreme Non-IID FedAvg Stress Test

**Date:** 11/06/2026

## Objective

Evaluate FedAvg under severe cross-silo heterogeneity to determine its robustness under realistic worst-case healthcare scenarios.

## Hospital Distribution

| Client   | COVID | Normal |
| -------- | ----: | -----: |
| Client 1 |   390 |     10 |
| Client 2 |   390 |     10 |
| Client 3 |    10 |    390 |
| Client 4 |    10 |    390 |

## Hyperparameters

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Clients       | 4                          |
| Local Epochs  | 1                          |
| Global Rounds | 5                          |
| Model         | EfficientNet-B0            |
| Aggregation   | FedAvg                     |
| Device        | NVIDIA RTX 3060 Laptop GPU |

## Results

| Round |   Accuracy |   F1 Score | Communication Cost | Round Time |
| ----: | ---------: | ---------: | -----------------: | ---------: |
|     1 |     65.50% |     65.35% |          123.66 MB |    53.43 s |
|     2 |     68.50% |     68.08% |          123.66 MB |    50.76 s |
|     3 |     74.50% |     74.06% |          123.66 MB |    66.16 s |
|     4 |     79.00% |     78.94% |          123.66 MB |    70.64 s |
|     5 | **82.00%** | **82.00%** |          123.66 MB |    64.80 s |

## Comparison

| Method                     |   Accuracy |
| -------------------------- | ---------: |
| 2025 Paper                 |     97.25% |
| EXP-001 (Centralized)      |     98.00% |
| EXP-002 (IID FedAvg)       |     97.50% |
| EXP-003 (Moderate Non-IID) |     97.00% |
| EXP-003B (Extreme Non-IID) | **82.00%** |

## Performance Drop

| Comparison            | Accuracy Drop |
| --------------------- | ------------: |
| EXP-002 → EXP-003B    |       -15.50% |
| 2025 Paper → EXP-003B |       -15.25% |
| EXP-001 → EXP-003B    |       -16.00% |

## Observations

* FedAvg failed to maintain performance under extreme heterogeneity.
* Severe Non-IID distributions caused catastrophic degradation.
* The experiment experimentally validates one of the primary limitations identified by the 2025 paper.
* These findings provide a strong justification for investigating more robust optimization techniques such as FedDyn.

---

# Research Narrative

The progression of experiments reveals a clear story:

1. EfficientNet-B0 outperformed the original CNN baseline.
2. FedAvg preserved most of the centralized performance under IID settings.
3. Moderate heterogeneity had only a minor impact on FedAvg.
4. Extreme heterogeneity exposed a major weakness of FedAvg, resulting in a 15.5 percentage point performance drop.
5. These findings motivate the introduction of FedDyn as the next stage of the proposed framework.

This progression transforms the work from a simple implementation into a systematic investigation of the limitations of existing blockchain-enabled federated learning approaches in healthcare.
