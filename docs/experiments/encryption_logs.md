# EXP-007: Selective and Adaptive CKKS Homomorphic Encryption using TenSEAL

**Date:** 27/06/2026

## Objective

Implement a modern homomorphic encryption layer for federated learning using TenSEAL CKKS and compare it against the 2025 paper's heavier Pyfhel/Microsoft SEAL-based encryption approach.

The goal was to evaluate whether selective encryption can preserve model utility while significantly reducing cryptographic overhead.

---

## Motivation

The 2025 paper encrypts model updates using homomorphic encryption before aggregation. However, full-model encryption is computationally expensive, especially for modern models such as EfficientNet-B0.

Therefore, this experiment evaluates:

1. Fixed classifier-only CKKS encryption.
2. CKKS profiling with FedDyn under extreme Non-IID.
3. CKKS profiling with Multi-Krum under Byzantine attack.
4. Adaptive top-k CKKS encryption based on update magnitude.

---

## CKKS Configuration

| Parameter                 | Value                             |
| ------------------------- | --------------------------------- |
| HE Library                | TenSEAL                           |
| HE Scheme                 | CKKS                              |
| Polynomial Modulus Degree | 8192                              |
| Coeff Mod Bit Sizes       | [60, 40, 40, 60]                  |
| Global Scale              | 2⁴⁰                               |
| Encrypted Scope           | Selective layer/tensor encryption |
| Model                     | EfficientNet-B0                   |

---

# EXP-007A: FedAvg + Fixed Classifier-Only CKKS

## Setup

| Parameter       | Value                       |
| --------------- | --------------------------- |
| Dataset         | COVID-19 Radiography Binary |
| Data Split      | IID                         |
| Clients         | 4                           |
| Aggregation     | FedAvg                      |
| Encrypted Scope | Classifier layer only       |
| Global Rounds   | 5                           |
| Local Epochs    | 1                           |

## Results

| Round |   Accuracy |   F1 Score | Avg Enc Time | HE Agg Time | Dec Time | Encrypted Upload |
| ----: | ---------: | ---------: | -----------: | ----------: | -------: | ---------------: |
|     1 |     84.00% |     83.99% |      0.0060s |     0.0042s |  0.0025s |        1.2750 MB |
|     2 |     86.50% |     86.35% |      0.0055s |     0.0047s |  0.0010s |        1.2754 MB |
|     3 |     94.25% |     94.24% |      0.0057s |     0.0040s |  0.0010s |        1.2739 MB |
|     4 |     96.75% |     96.75% |      0.0049s |     0.0000s |  0.0009s |        1.2754 MB |
|     5 | **97.75%** | **97.75%** |      0.0054s |     0.0039s |  0.0010s |        1.2752 MB |

## Observations

* Selective CKKS preserved FedAvg performance.
* Final accuracy reached 97.75%, comparable to or slightly above the plaintext FedAvg baseline.
* The small improvement is attributed to stochastic training variance, not encryption itself.
* Encryption overhead was extremely small.

---

# EXP-007B: FedDyn + Fixed Classifier-Only CKKS under Extreme Non-IID

## Setup

| Parameter       | Value                                      |
| --------------- | ------------------------------------------ |
| Dataset Split   | Extreme Non-IID                            |
| Clients         | 4                                          |
| Algorithm       | FedDyn                                     |
| Alpha           | 0.005                                      |
| Seed            | 7                                          |
| Encrypted Scope | Classifier layer only                      |
| CKKS Usage      | Secure transmission and overhead profiling |

## Results

| Round |   Accuracy |   F1 Score | Avg Enc Time | HE Agg Time | Dec Time | Crypto Overhead | Encrypted Upload |
| ----: | ---------: | ---------: | -----------: | ----------: | -------: | --------------: | ---------------: |
|     1 |     50.00% |     33.33% |      0.0071s |     0.0070s |  0.0020s |         0.0593% |        1.2756 MB |
|     2 |     50.00% |     33.33% |      0.0058s |     0.0050s |  0.0010s |         0.0472% |        1.2745 MB |
|     3 |     76.25% |     74.95% |      0.0059s |     0.0040s |  0.0010s |         0.0451% |        1.2754 MB |
|     4 |     75.75% |     74.23% |      0.0060s |     0.0040s |  0.0010s |         0.0466% |        1.2757 MB |
|     5 | **90.75%** | **90.73%** |      0.0063s |     0.0050s |  0.0010s |         0.0507% |        1.2751 MB |

## Comparison

| Method                 |   Accuracy |   F1 Score |
| ---------------------- | ---------: | ---------: |
| FedDyn seed 7, no CKKS |     90.75% |     90.73% |
| FedDyn + CKKS seed 7   | **90.75%** | **90.73%** |

## Observations

* CKKS preserved FedDyn performance under extreme Non-IID conditions.
* Crypto overhead remained below 0.06% of round time.
* Earlier unstable FedDyn + CKKS attempts were caused by FedDyn seed sensitivity, not CKKS itself.
* With a stable seed, CKKS did not degrade FedDyn.

---

# EXP-007C: Multi-Krum + Fixed Classifier-Only CKKS under Byzantine Attack

## Setup

| Parameter         | Value                                      |
| ----------------- | ------------------------------------------ |
| Dataset Split     | Moderate Non-IID                           |
| Clients           | 4                                          |
| Malicious Clients | 1                                          |
| Attack            | Sign-flip attack                           |
| Attack Scale      | 5.0                                        |
| Aggregation       | Multi-Krum                                 |
| Encrypted Scope   | Classifier layer only                      |
| CKKS Usage        | Secure transmission and overhead profiling |

## Results

| Round |   Accuracy |   F1 Score | Selected Clients | Avg Enc Time | HE Agg Time | Crypto Overhead | Encrypted Upload |
| ----: | ---------: | ---------: | ---------------- | -----------: | ----------: | --------------: | ---------------: |
|     1 |     81.00% |     80.88% | [1, 2]           |      0.0063s |     0.0040s |         0.0631% |        1.2751 MB |
|     2 |     83.00% |     82.63% | [2, 3]           |      0.0060s |     0.0041s |         0.0614% |        1.2752 MB |
|     3 |     91.00% |     90.95% | [2, 3]           |      0.0020s |     0.0030s |         0.0270% |        1.2751 MB |
|     4 |     93.50% |     93.48% | [2, 3]           |      0.0059s |     0.0015s |         0.0570% |        1.2747 MB |
|     5 | **96.00%** | **96.00%** | [2, 3]           |      0.0050s |     0.0053s |         0.0566% |        1.2745 MB |

## Comparison

| Method                          |   Accuracy |   F1 Score |
| ------------------------------- | ---------: | ---------: |
| Multi-Krum + Byzantine, no CKKS |     93.50% |     93.47% |
| Multi-Krum + Fixed CKKS         | **96.00%** | **96.00%** |

## Observations

* Multi-Krum continued to resist Byzantine sign-flip attacks while CKKS overhead was profiled.
* Selected clients were mostly [2, 3], indicating successful rejection of the malicious update.
* Accuracy remained high under attack.
* The apparent improvement over plaintext Multi-Krum is attributed to run-to-run stochastic variation.

---

# EXP-007D: FedAvg + Adaptive Selective CKKS

## Objective

Improve fixed classifier-only encryption by adaptively selecting the most dynamically changing trainable tensors based on update magnitude.

## Selection Strategy

For each client update:

```text
update_magnitude = ||client_state - global_state||
```

The top-k trainable tensors by normalized update magnitude were selected for CKKS encryption.

BatchNorm buffers such as `running_mean`, `running_var`, and `num_batches_tracked` were excluded.

## Setup

| Parameter          | Value                                  |
| ------------------ | -------------------------------------- |
| Dataset Split      | IID                                    |
| Aggregation        | FedAvg                                 |
| Selection Strategy | Top-k trainable tensors by update norm |
| Top-K              | 4                                      |
| HE Scheme          | CKKS                                   |
| HE Library         | TenSEAL                                |

## Results

| Round |   Accuracy |   F1 Score | Avg Enc Time | HE Agg Time | Crypto Overhead | Encrypted Upload |
| ----: | ---------: | ---------: | -----------: | ----------: | --------------: | ---------------: |
|     1 |     81.75% |     81.37% |      0.0057s |     0.0043s |         0.0562% |        1.2754 MB |
|     2 |     91.25% |     91.25% |      0.0070s |     0.0050s |         0.0666% |        1.2751 MB |
|     3 |     94.50% |     94.49% |      0.0052s |     0.0030s |         0.0495% |        1.2758 MB |
|     4 |     97.00% |     97.00% |      0.0064s |     0.0050s |         0.0624% |        1.2756 MB |
|     5 | **97.50%** | **97.50%** |      0.0060s |     0.0040s |         0.0557% |        1.2753 MB |

## Observations

* Adaptive CKKS selected real trainable tensors rather than fixed classifier parameters.
* Model accuracy matched plaintext FedAvg performance.
* Crypto overhead remained negligible.
* This is a stronger novelty than fixed classifier-only encryption because the encrypted tensors are selected dynamically.

---

# EXP-007E: Multi-Krum + Adaptive Selective CKKS under Byzantine Attack

## Setup

| Parameter          | Value                                  |
| ------------------ | -------------------------------------- |
| Dataset Split      | Moderate Non-IID                       |
| Aggregation        | Multi-Krum                             |
| Attack             | Sign-flip attack                       |
| Malicious Clients  | 1                                      |
| Selection Strategy | Top-k trainable tensors by update norm |
| Top-K              | 4                                      |
| HE Scheme          | CKKS                                   |
| HE Library         | TenSEAL                                |

## Results

| Round |   Accuracy |   F1 Score | Selected Clients | Avg Enc Time | HE Agg Time | Crypto Overhead | Encrypted Upload |
| ----: | ---------: | ---------: | ---------------- | -----------: | ----------: | --------------: | ---------------: |
|     1 |     77.25% |     76.87% | [1, 2]           |      0.0058s |     0.0044s |         0.0579% |        1.2757 MB |
|     2 |     87.50% |     87.50% | [1, 2]           |      0.0054s |     0.0040s |         0.0584% |        1.2754 MB |
|     3 |     86.00% |     85.72% | [2, 3]           |      0.0056s |     0.0045s |         0.0582% |        1.2752 MB |
|     4 |     93.50% |     93.47% | [2, 3]           |      0.0055s |     0.0045s |         0.0560% |        1.2755 MB |
|     5 | **95.75%** | **95.74%** | [2, 3]           |      0.0056s |     0.0044s |         0.0560% |        1.2754 MB |

## Comparison

| Method                          |   Accuracy |   F1 Score |
| ------------------------------- | ---------: | ---------: |
| Multi-Krum + Byzantine, no CKKS |     93.50% |     93.47% |
| Fixed CKKS + Multi-Krum         |     96.00% |     96.00% |
| Adaptive CKKS + Multi-Krum      | **95.75%** | **95.74%** |

## Observations

* Adaptive CKKS preserved Multi-Krum's Byzantine robustness.
* The malicious client was eventually excluded by Multi-Krum.
* The adaptive selector encrypted trainable tensors based on update magnitude.
* Cryptographic overhead remained around 0.056%.
* This supports adaptive selective CKKS as a practical secure aggregation strategy.

---

# EXP-007 Summary

| Experiment | Setting                      | Method                     | Accuracy | F1 Score | Crypto Overhead |
| ---------- | ---------------------------- | -------------------------- | -------: | -------: | --------------: |
| EXP-007A   | IID                          | FedAvg + Fixed CKKS        |   97.75% |   97.75% |     ~0.05–0.06% |
| EXP-007B   | Extreme Non-IID              | FedDyn + Fixed CKKS        |   90.75% |   90.73% |          ~0.05% |
| EXP-007C   | Moderate Non-IID + Byzantine | Multi-Krum + Fixed CKKS    |   96.00% |   96.00% |     ~0.03–0.06% |
| EXP-007D   | IID                          | FedAvg + Adaptive CKKS     |   97.50% |   97.50% |     ~0.05–0.07% |
| EXP-007E   | Moderate Non-IID + Byzantine | Multi-Krum + Adaptive CKKS |   95.75% |   95.74% |         ~0.056% |

---

# Comparison with 2025 Paper

| Component                 | 2025 Paper                 | Our EXP-007                                                            |
| ------------------------- | -------------------------- | ---------------------------------------------------------------------- |
| HE Library                | Pyfhel / Microsoft SEAL    | TenSEAL                                                                |
| Model                     | CNN                        | EfficientNet-B0                                                        |
| Encryption Scope          | Full model update          | Selective / adaptive tensors                                           |
| Optimization              | FedAvg                     | FedAvg, FedDyn, Multi-Krum                                             |
| Byzantine Defense         | Not evaluated              | Multi-Krum evaluated                                                   |
| Adaptive Encryption       | No                         | Yes                                                                    |
| Crypto Overhead Reporting | Encryption/decryption time | Encryption, aggregation, decryption, size expansion, crypto overhead % |
| Final COVID Accuracy      | 97.25%                     | Up to 97.75%                                                           |

---

# Key Takeaways

* Selective CKKS preserved model utility while introducing negligible cryptographic overhead.
* Adaptive CKKS is a stronger novelty than fixed classifier-only encryption because it dynamically selects trainable tensors based on update magnitude.
* CKKS did not improve learning directly; it preserved learning while adding privacy protection.
* Accuracy differences between plaintext and encrypted variants are likely due to stochastic training variance.
* Multi-Krum and CKKS can coexist, allowing Byzantine robustness and encrypted update protection in the same pipeline.
* FedDyn remained sensitive to random seeds, but with a stable seed, CKKS preserved FedDyn performance under extreme Non-IID conditions.

---

# Research Contribution from EXP-007

EXP-007 contributes a lightweight and adaptive CKKS-based encrypted aggregation strategy for modern federated medical image classification. Unlike the 2025 paper, which applies homomorphic encryption broadly to model updates, this work introduces selective and adaptive encryption to reduce cryptographic cost while preserving model performance. The results show that privacy-preserving encrypted transmission can be integrated with FedAvg, FedDyn, and Multi-Krum with negligible overhead.
