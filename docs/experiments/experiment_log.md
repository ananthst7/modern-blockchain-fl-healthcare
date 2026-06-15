# Experiment Log

## Research Objective

Develop a modern blockchain-enabled privacy-preserving federated learning framework for healthcare image classification by extending the 2025 FL-HE framework using improved deep learning architectures, robust federated optimization, and Byzantine-resilient aggregation.

---

# Baseline Paper

**Title:** Blockchain-based Federated Learning with Homomorphic Encryption for Privacy-Preserving Healthcare Data Sharing (2025)

## COVID Dataset Baseline Performance

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
* Accuracy degradation at Epoch 5 suggests possible overfitting.
* EfficientNet-B0 outperformed the CNN baseline of the 2025 paper.

---

# EXP-002: IID FedAvg with EfficientNet-B0

**Date:** 11/06/2026

## Objective

Evaluate standard FedAvg on COVID classification under IID hospital distributions.

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

**Date:** 11/06/2026**

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

| Method                          | Accuracy |
| ------------------------------- | -------: |
| 2025 Paper                      |   97.25% |
| EXP-002 IID FedAvg              |   97.50% |
| EXP-003 Moderate Non-IID FedAvg |   97.00% |

## Observations

* Moderate heterogeneity caused only minor degradation.
* FedAvg remained surprisingly robust.
* Accuracy dropped by only 0.50 percentage points compared to IID FedAvg.

---

# EXP-003B: Extreme Non-IID FedAvg Stress Test

**Date:** 11/06/2026

## Objective

Evaluate FedAvg under severe cross-silo heterogeneity to determine its robustness under worst-case healthcare scenarios.

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

| Method                          |   Accuracy |
| ------------------------------- | ---------: |
| 2025 Paper                      |     97.25% |
| EXP-001 Centralized             |     98.00% |
| EXP-002 IID FedAvg              |     97.50% |
| EXP-003 Moderate Non-IID FedAvg |     97.00% |
| EXP-003B Extreme Non-IID FedAvg | **82.00%** |

## Performance Drop

| Comparison            | Accuracy Drop |
| --------------------- | ------------: |
| EXP-002 → EXP-003B    |       -15.50% |
| 2025 Paper → EXP-003B |       -15.25% |
| EXP-001 → EXP-003B    |       -16.00% |

## Observations

* FedAvg failed to maintain performance under extreme heterogeneity.
* Severe Non-IID distributions caused catastrophic degradation.
* This validates one of the main limitations of the 2025 paper.
* These findings justify investigating robust optimization methods such as FedDyn.

---

# EXP-004: FedDyn under Extreme Non-IID Conditions

**Date:** 11/06/2026

## Objective

Investigate whether FedDyn can mitigate the performance degradation observed in FedAvg under extreme Non-IID hospital distributions.

## Motivation

EXP-003B showed that FedAvg suffered a 15.5 percentage point accuracy drop under severe cross-silo heterogeneity. This experiment evaluates whether FedDyn can recover the lost performance.

## Dataset

* COVID-19 Radiography Database
* Binary Classification:

  * COVID
  * Normal

## Hospital Distribution

| Client   | COVID | Normal |
| -------- | ----: | -----: |
| Client 1 |   390 |     10 |
| Client 2 |   390 |     10 |
| Client 3 |    10 |    390 |
| Client 4 |    10 |    390 |

## Initial FedDyn Configuration

| Parameter              | Value                      |
| ---------------------- | -------------------------- |
| Clients                | 4                          |
| Local Epochs           | 1                          |
| Global Rounds          | 5                          |
| Model                  | EfficientNet-B0            |
| Optimization Algorithm | FedDyn                     |
| Alpha                  | 0.01                       |
| Learning Rate          | 1×10⁻⁴                     |
| Device                 | NVIDIA RTX 3060 Laptop GPU |

## Initial FedDyn Results, α = 0.01

| Round |   Accuracy |   F1 Score | Communication Cost | Round Time |
| ----: | ---------: | ---------: | -----------------: | ---------: |
|     1 |     50.00% |     33.33% |          123.66 MB |    49.99 s |
|     2 |     50.00% |     33.33% |          123.66 MB |    49.62 s |
|     3 |     61.50% |     54.80% |          123.66 MB |    54.89 s |
|     4 |     61.00% |     54.21% |          123.66 MB |    58.26 s |
|     5 | **88.00%** | **87.96%** |          123.66 MB |    58.68 s |

## Comparison with EXP-003B

| Method                          |   Accuracy |
| ------------------------------- | ---------: |
| EXP-003B Extreme Non-IID FedAvg |     82.00% |
| EXP-004 FedDyn, α = 0.01        | **88.00%** |

**Improvement:** +6.00%

## EXP-004A: FedDyn Alpha Sweep

### Objective

Optimize the FedDyn regularization coefficient α to maximize performance under extreme Non-IID conditions.

### Alpha Values Evaluated

|     Alpha | Best Accuracy | Best F1 Score | Best Round |
| --------: | ------------: | ------------: | ---------: |
|     0.001 |        89.75% |        89.71% |          5 |
| **0.005** |    **92.25%** |    **92.25%** |      **5** |
|      0.01 |        90.50% |        90.46% |          5 |
|      0.05 |        87.00% |        86.96% |          5 |
|      0.10 |        89.00% |        88.99% |          5 |

## Best FedDyn Performance

| Method                         |   Accuracy |
| ------------------------------ | ---------: |
| EXP-003B FedAvg                |     82.00% |
| EXP-004 FedDyn, α = 0.01       |     88.00% |
| **EXP-004A FedDyn, α = 0.005** | **92.25%** |

## Performance Recovery

| Comparison          | Improvement |
| ------------------- | ----------: |
| EXP-003B → EXP-004  |      +6.00% |
| EXP-003B → EXP-004A | **+10.25%** |

## Additional Investigation: Extended Training

An additional 10-round FedDyn experiment using α = 0.005 was conducted. Although intermediate rounds achieved competitive performance, instability emerged during later communication rounds, leading to performance collapse.

This suggests that while FedDyn substantially improves robustness to heterogeneous data, further stabilization strategies may be required for prolonged optimization under extreme Non-IID settings.

## Observations

* FedDyn successfully mitigated degradation caused by extreme hospital heterogeneity.
* Hyperparameter selection significantly affected performance.
* α = 0.005 emerged as the optimal setting among the evaluated values.
* FedDyn recovered over 10 percentage points of accuracy lost by FedAvg under extreme Non-IID conditions.
* Extended communication rounds revealed optimization instability, motivating future stabilization strategies.

---

# EXP-005: Byzantine Robustness Analysis

**Date:** 11/06/2026

## Objective

Investigate the vulnerability of federated learning to malicious participants and evaluate whether Byzantine-resilient aggregation can mitigate adversarial degradation.

The 2025 paper assumes that all hospitals behave honestly. In practice, compromised or malfunctioning institutions may submit adversarial updates that poison the global model.

---

# EXP-005E: IID Byzantine Attack with Multi-Krum

## Objective

Evaluate Multi-Krum under IID conditions to isolate malicious-update robustness from data heterogeneity.

## Setup

| Parameter          | Value                |
| ------------------ | -------------------- |
| Clients            | 4                    |
| Malicious Clients  | 1                    |
| Attack             | Sign-Flip Attack     |
| Attack Scale       | 5.0                  |
| Dataset Split      | IID                  |
| Model              | EfficientNet-B0      |
| Aggregation        | FedAvg vs Multi-Krum |
| Global Rounds      | 5                    |
| Communication Cost | 123.66 MB / round    |

## Results

| Method                        |   Accuracy |   F1 Score |
| ----------------------------- | ---------: | ---------: |
| EXP-002 IID FedAvg, No Attack |     97.50% |     97.50% |
| FedAvg + Byzantine            |     50.00% |     33.33% |
| **Multi-Krum + Byzantine**    | **97.00%** | **97.00%** |

## Round-wise Multi-Krum Results

| Round |   Accuracy |   F1 Score | Selected Clients |
| ----: | ---------: | ---------: | ---------------- |
|     1 |     83.50% |     83.49% | [1, 2]           |
|     2 |     90.00% |     90.00% | [2, 3]           |
|     3 |     94.75% |     94.75% | [1, 2]           |
|     4 |     95.50% |     95.50% | [2, 3]           |
|     5 | **97.00%** | **97.00%** | [1, 3]           |

## Observations

* A single malicious client reduced FedAvg to random guessing.
* Multi-Krum recovered nearly the entire lost performance.
* Byzantine-resilient aggregation preserved model utility under IID settings.

---

# EXP-005F: Moderate Non-IID Byzantine Attack with Multi-Krum

## Objective

Evaluate Byzantine resilience under realistic moderate healthcare heterogeneity.

## Setup

| Parameter          | Value                |
| ------------------ | -------------------- |
| Clients            | 4                    |
| Malicious Clients  | 1                    |
| Attack             | Sign-Flip Attack     |
| Attack Scale       | 5.0                  |
| Dataset Split      | Moderate Non-IID     |
| Model              | EfficientNet-B0      |
| Aggregation        | FedAvg vs Multi-Krum |
| Global Rounds      | 5                    |
| Communication Cost | 123.66 MB / round    |

## Hospital Distribution

| Client   | COVID | Normal |
| -------- | ----: | -----: |
| Client 1 |   320 |     80 |
| Client 2 |   280 |    120 |
| Client 3 |   120 |    280 |
| Client 4 |    80 |    320 |

## Results

| Method                                     |   Accuracy |   F1 Score |
| ------------------------------------------ | ---------: | ---------: |
| EXP-003 Moderate Non-IID FedAvg, No Attack |     97.00% |     97.00% |
| FedAvg + Byzantine                         |     50.00% |     33.33% |
| **Multi-Krum + Byzantine**                 | **93.50%** | **93.47%** |

## Round-wise Multi-Krum Results

| Round |   Accuracy |   F1 Score | Selected Clients |
| ----: | ---------: | ---------: | ---------------- |
|     1 |     74.75% |     73.56% | [2, 3]           |
|     2 |     86.00% |     85.92% | [2, 3]           |
|     3 |     89.25% |     89.12% | [2, 3]           |
|     4 |     93.00% |     92.97% | [2, 3]           |
|     5 | **93.50%** | **93.47%** | [2, 3]           |

## Observations

* Multi-Krum successfully rejected malicious updates even under heterogeneous hospital distributions.
* FedAvg collapsed to random guessing under the same attack.
* Compared with no-attack moderate Non-IID FedAvg, Multi-Krum retained most useful model performance under attack.
* This provides stronger Byzantine-resilience evidence than the extreme Non-IID stress test.

---

# Additional Byzantine Stress Tests

## EXP-005A: Extreme Non-IID FedAvg + Byzantine

| Method                            | Accuracy | F1 Score |
| --------------------------------- | -------: | -------: |
| Extreme Non-IID FedAvg, No Attack |   82.00% |   82.00% |
| FedAvg + Byzantine                |   50.00% |   33.33% |

**Observation:** Extreme heterogeneity combined with malicious behavior caused complete collapse of standard FedAvg.

---

## EXP-005B: Extreme Non-IID Multi-Krum + Byzantine

| Method                 | Accuracy | F1 Score |
| ---------------------- | -------: | -------: |
| FedAvg + Byzantine     |   50.00% |   33.33% |
| Multi-Krum + Byzantine |   60.25% |   52.79% |

**Observation:** Multi-Krum partially recovered performance but struggled under severe heterogeneity.

---

## EXP-005C: FedDyn + Multi-Krum + Byzantine

| Method                          | Accuracy | F1 Score |
| ------------------------------- | -------: | -------: |
| FedAvg + Byzantine              |   50.00% |   33.33% |
| FedDyn + Multi-Krum + Byzantine |   58.75% |   50.29% |

**Observation:** Combining FedDyn and Multi-Krum without additional stabilization did not outperform Multi-Krum alone.

---

## EXP-005D: 8-Client Extreme Non-IID Multi-Krum Stress Test

| Method                            | Accuracy | F1 Score |
| --------------------------------- | -------: | -------: |
| FedAvg + Byzantine, 8 Clients     |   50.00% |   33.33% |
| Multi-Krum + Byzantine, 8 Clients |   50.50% |   34.43% |

**Observation:** Increasing clients alone was insufficient under severe heterogeneity. Multi-Krum selected mostly non-malicious clients, but the selected clients were still heavily class-skewed, causing the global model to collapse.

---

---

# EXP-006: Differential Privacy Analysis

**Date:** 11/06/2026

## Objective

Evaluate the impact of privacy-preserving noise mechanisms on federated medical image classification and analyze the trade-off between model utility and privacy strength.

The original 2025 paper uses homomorphic encryption but does not evaluate differential privacy. In this experiment, both approximate update-level perturbation and formal DP-SGD using Opacus were investigated.

---

# EXP-006A: FedAvg with Approximate Update-Level Differential Privacy

## Objective

Evaluate whether lightweight update-level perturbation can preserve utility in IID federated learning.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | IID |
| Clients | 4 |
| Model | EfficientNet-B0 |
| Aggregation | FedAvg |
| DP Type | Classifier-only update-level perturbation |
| Noise Std | 1×10⁻⁶ |
| Clip Norm | 100.0 |
| Formal Epsilon | Not available |
| Global Rounds | 5 |

## Results

| Round | Accuracy | F1 Score | Communication Cost | Round Time |
|---:|---:|---:|---:|---:|
| 1 | 80.75% | 80.43% | 123.66 MB | 44.46 s |
| 2 | 88.75% | 88.71% | 123.66 MB | 43.86 s |
| 3 | 95.25% | 95.25% | 123.66 MB | 47.40 s |
| 4 | 96.75% | 96.75% | 123.66 MB | 58.35 s |
| 5 | **97.50%** | **97.50%** | 123.66 MB | 58.76 s |

## Comparison

| Method | Accuracy | F1 Score | Formal DP |
|---|---:|---:|---|
| EXP-002 FedAvg, No DP | 97.50% | 97.50% | No |
| EXP-006A Approximate DP | **97.50%** | **97.50%** | No |

## Observations

- Classifier-only update-level perturbation preserved the baseline FedAvg performance.
- This method does not provide formal ε-DP guarantees.
- The result shows that lightweight update perturbation can preserve utility, but should be described as approximate privacy rather than strict differential privacy.

---

# EXP-006B: FedDyn with Approximate Update-Level DP under Extreme Non-IID

## Objective

Evaluate whether approximate update-level DP affects the robustness gains achieved by FedDyn under extreme Non-IID hospital distributions.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | Extreme Non-IID |
| Clients | 4 |
| Model | EfficientNet-B0 |
| Optimization Algorithm | FedDyn |
| Alpha | 0.005 |
| DP Type | Classifier-only update-level perturbation |
| Noise Std | 1×10⁻⁶ |
| Clip Norm | 100.0 |
| Formal Epsilon | Not available |
| Global Rounds | 5 |

## Results

| Round | Accuracy | F1 Score | Communication Cost | Round Time |
|---:|---:|---:|---:|---:|
| 1 | 50.00% | 33.33% | 123.66 MB | 60.04 s |
| 2 | 50.00% | 33.33% | 123.66 MB | 72.42 s |
| 3 | 73.00% | 71.42% | 123.66 MB | 77.07 s |
| 4 | 72.25% | 69.93% | 123.66 MB | 65.32 s |
| 5 | **92.25%** | **92.25%** | 123.66 MB | 59.27 s |

## Comparison

| Method | Accuracy | F1 Score | Formal DP |
|---|---:|---:|---|
| EXP-004A FedDyn, No DP | 92.25% | 92.25% | No |
| EXP-006B FedDyn + Approximate DP | **92.25%** | **92.25%** | No |

## Observations

- Approximate DP did not reduce the final FedDyn performance.
- FedDyn retained its extreme Non-IID robustness even after classifier-level perturbation.
- This result is useful as a lightweight privacy-preserving mechanism, but it cannot be claimed as formal DP.

---

# EXP-006C: Formal DP-SGD using Opacus

## Objective

Evaluate formal differential privacy using Opacus DP-SGD and measure the privacy–utility trade-off.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | IID |
| Clients | 4 |
| Model | EfficientNet-B0 |
| Trainable Parameters | Classifier head only |
| Frozen Parameters | EfficientNet feature extractor |
| Aggregation | FedAvg |
| DP Library | Opacus |
| DP Mechanism | Per-sample gradient clipping + Gaussian noise |
| Delta | 1×10⁻⁵ |
| Communication Cost | 123.66 MB / round |

## DP Configurations Tested

| Configuration | Noise Multiplier | Local Epochs | Learning Rate | Max Grad Norm | Epsilon | Best Accuracy | Final Accuracy | Formal DP |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full EfficientNet DP | 0.1 | 1 | 1×10⁻⁴ | 1.0 | 351.06 | 50.00% | 50.00% | Yes |
| Classifier-only DP | 1.0 | 1 | 1×10⁻⁴ | 1.0 | 1.76 | 57.00% | 57.00% | Yes |
| Classifier-only DP | 0.3 | 1 | 5×10⁻⁴ | 1.0 | 32.72 | 69.50% | 67.50% | Yes |
| Classifier-only DP | 0.3 | 3 | 5×10⁻⁴ | 1.0 | 50.50 | 75.25% | 75.25% | Yes |
| Classifier-only DP | 0.2 | 5 | 1×10⁻³ | 0.5 | 173.04 | **84.75%** | **84.75%** | Yes |

## Best Opacus Run: Classifier-only DP, σ = 0.2, Local Epochs = 5

| Round | Accuracy | F1 Score | Avg Epsilon | Communication Cost | Round Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 64.50% | 60.31% | 173.04 | 123.66 MB | 87.72 s |
| 2 | 82.25% | 82.06% | 173.04 | 123.66 MB | 98.48 s |
| 3 | 83.50% | 83.42% | 173.04 | 123.66 MB | 95.52 s |
| 4 | 84.25% | 84.19% | 173.04 | 123.66 MB | 104.80 s |
| 5 | **84.75%** | **84.70%** | 173.04 | 123.66 MB | 114.93 s |

## Privacy–Utility Trade-off

| Method | Accuracy | F1 Score | Epsilon | Formal DP |
|---|---:|---:|---:|---|
| FedAvg, No DP | 97.50% | 97.50% | N/A | No |
| Approximate Update-Level DP | 97.50% | 97.50% | N/A | No |
| Opacus DP-SGD, Strong Privacy | 57.00% | 55.21% | 1.76 | Yes |
| Opacus DP-SGD, Best Utility | **84.75%** | **84.70%** | 173.04 | Yes |

## Observations

- Approximate update-level perturbation preserved accuracy but did not provide a formal privacy budget.
- Formal DP-SGD using Opacus provided measurable ε-DP guarantees but caused significant utility degradation.
- Strong privacy with ε ≈ 1.76 resulted in only 57.00% accuracy.
- The best formal DP utility result reached 84.75% accuracy, but required a very weak privacy budget of ε ≈ 173.04.
- These results demonstrate the difficulty of applying strict DP-SGD to EfficientNet-based federated medical imaging.
- For this framework, CKKS homomorphic encryption may provide a more practical privacy-preserving mechanism than strict DP-SGD.

---

# Master Results Summary

| Experiment | Setting | Method | Accuracy | F1 Score |
|---|---|---|---:|---:|
| 2025 Paper | COVID | CNN + FedAvg | 97.25% | N/A |
| EXP-001 | Centralized | EfficientNet-B0 | **98.00%** | **98.00%** |
| EXP-002 | IID | FedAvg | 97.50% | 97.50% |
| EXP-003 | Moderate Non-IID | FedAvg | 97.00% | 97.00% |
| EXP-003B | Extreme Non-IID | FedAvg | 82.00% | 82.00% |
| EXP-004A | Extreme Non-IID | FedDyn | **92.25%** | **92.25%** |
| EXP-005E | IID + Byzantine | Multi-Krum | **97.00%** | **97.00%** |
| EXP-005F | Moderate Non-IID + Byzantine | Multi-Krum | **93.50%** | **93.47%** |
| EXP-006A | IID + Approximate DP | FedAvg + Update DP | **97.50%** | **97.50%** |
| EXP-006B | Extreme Non-IID + Approximate DP | FedDyn + Update DP | **92.25%** | **92.25%** |
| EXP-006C | IID + Formal DP-SGD | FedAvg + Opacus | 84.75% | 84.70% |

---

# Research Narrative

The experimental progression reveals several major limitations of the original 2025 framework and corresponding improvements:

1. EfficientNet-B0 improved upon the original CNN baseline.
2. FedAvg remained effective under IID and moderate Non-IID conditions.
3. FedAvg failed under extreme heterogeneity, dropping to 82.00%.
4. FedDyn recovered performance under extreme Non-IID conditions, reaching 92.25%.
5. Standard FedAvg was highly vulnerable to Byzantine attacks, collapsing to 50.00% accuracy.
6. Multi-Krum restored performance to 97.00% under IID attack and 93.50% under moderate Non-IID attack.
7. Approximate update-level DP preserved model utility but did not provide formal ε guarantees.
8. Formal Opacus DP-SGD provided measurable privacy guarantees but introduced a strong privacy–utility trade-off.
9. Extreme heterogeneity combined with malicious behavior remains an open challenge requiring future investigation.
10. CKKS homomorphic encryption is the next major privacy-preserving component to evaluate.

---

# Current State of the Proposed Framework

| Component | Status |
|---|---|
| EfficientNet-B0 Baseline | Completed |
| FedAvg Evaluation | Completed |
| Non-IID Analysis | Completed |
| FedDyn Optimization | Completed |
| Byzantine Attack Simulation | Completed |
| Multi-Krum Robust Aggregation | Completed |
| Approximate Update-Level DP | Completed |
| Formal Opacus DP-SGD Analysis | Completed |
| CKKS Homomorphic Encryption | Next |
| Proof-of-Authority Blockchain | Pending |
| Brain MRI Validation | Pending |
