# EXP-007: Selective, Adaptive, and Budgeted CKKS Homomorphic Encryption using TenSEAL

## Objective

Implement and evaluate a modern CKKS homomorphic encryption layer for the proposed federated healthcare framework using TenSEAL.

The objective of EXP-007 is not only to reproduce the homomorphic encryption component of the 2025 paper, but to extend it with:

1. Selective CKKS encryption.
2. Adaptive tensor selection.
3. Budget-constrained adaptive encryption.
4. Compatibility analysis with FedAvg, FedDyn, and Multi-Krum.
5. Cryptographic overhead profiling.
6. Privacy-coverage metrics for adaptive encryption.

---

## Motivation

The 2025 paper uses homomorphic encryption to protect model updates during federated aggregation. However, encrypting large model updates is expensive, especially when replacing the original CNN with a modern EfficientNet-B0 backbone.

To improve practicality, this work evaluates selective CKKS encryption instead of encrypting the full model. The experiments investigate whether selected model tensors can be encrypted while preserving learning utility and reducing cryptographic overhead.

---

## Difference from the 2025 Paper

| Component | 2025 Paper | EXP-007 Implementation |
|---|---|---|
| HE Library | Pyfhel / Microsoft SEAL | TenSEAL |
| Model | CNN | EfficientNet-B0 |
| FL Algorithm | FedAvg | FedAvg, FedDyn, Multi-Krum |
| Encryption Strategy | Homomorphic encryption over model updates | Fixed selective, adaptive selective, and budgeted adaptive CKKS |
| Byzantine Robustness | Not evaluated | Evaluated using Multi-Krum under sign-flip attack |
| Adaptive Encryption | Not present | Top-k trainable tensor selection by update magnitude |
| Budget-Aware Encryption | Not present | 20 KB lightweight and 1 MB privacy-focused budgets |
| Privacy-Coverage Metrics | Not present | UCR / ICR, PER, AEQ, ILR / RRS |
| Systems Profiling | Limited | Encryption time, decryption time, HE aggregation time, ciphertext expansion, encrypted upload, crypto overhead |

---

## CKKS Configuration

| Parameter | Value |
|---|---|
| HE Library | TenSEAL |
| HE Scheme | CKKS |
| Polynomial Modulus Degree | 8192 |
| Coeff Mod Bit Sizes | [60, 40, 40, 60] |
| Global Scale | 2⁴⁰ |
| Model | EfficientNet-B0 |
| Dataset | COVID-19 Radiography Binary |
| Clients | 4 |
| Global Rounds | 5 |
| Local Epochs | 1 |

---

## Important Interpretation

CKKS encryption does **not** improve model learning directly. Any small accuracy increase observed between encrypted and plaintext runs is attributed to stochastic training variation.

The correct claim is:

> CKKS preserved model utility while protecting selected model updates and adding measurable but manageable cryptographic overhead.

---

# Adaptive Encryption Quality Metrics

Because CKKS does not provide an epsilon-like privacy score like differential privacy, this work introduces selection-quality and privacy-coverage metrics to evaluate whether adaptive encryption is choosing meaningful tensors.

## Metrics

| Metric | Formula / Meaning | Interpretation |
|---|---|---|
| Update Coverage Ratio / Information Coverage Ratio (UCR / ICR) | Encrypted update norm / total update norm | Fraction of update movement protected by encryption |
| Parameter Encryption Ratio (PER) | Encrypted parameters / total trainable parameters | Fraction of model parameters encrypted |
| Adaptive Encryption Quality (AEQ) | UCR / PER | Efficiency of adaptive encryption |
| Information Leakage Ratio (ILR) | 1 − UCR | Fraction of update movement still visible |
| Reconstruction Risk Score (RRS) | Same as ILR | Approximate remaining reconstruction exposure |
| Adaptive Gain | UCR(adaptive) − UCR(fixed classifier) | Improvement over fixed classifier-only encryption |
| Crypto Overhead | Crypto time / round time | Practical deployment cost |
| Encrypted Upload | Total ciphertext upload per round | Communication overhead |

## Why These Metrics Matter

CKKS provides cryptographic confidentiality for encrypted tensors, but selective encryption raises a second question:

> Are the selected tensors important enough to justify encrypting only part of the model?

The adaptive metrics answer this by measuring how much of the model update signal is protected relative to how many parameters are encrypted.

---

# EXP-007A: FedAvg + Fixed Classifier-Only CKKS

## Objective

Evaluate whether encrypting only the classifier layer preserves IID FedAvg performance while reducing homomorphic encryption overhead.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | IID |
| Aggregation | FedAvg |
| Encrypted Scope | Classifier layer only |
| Selection Type | Fixed |
| Clients | 4 |
| Global Rounds | 5 |

## Results

| Round | Accuracy | F1 Score | Avg Enc Time | HE Agg Time | Dec Time | Encrypted Upload |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.00% | 83.99% | 0.0060s | 0.0042s | 0.0025s | 1.2750 MB |
| 2 | 86.50% | 86.35% | 0.0055s | 0.0047s | 0.0010s | 1.2754 MB |
| 3 | 94.25% | 94.24% | 0.0057s | 0.0040s | 0.0010s | 1.2739 MB |
| 4 | 96.75% | 96.75% | 0.0049s | 0.0000s | 0.0009s | 1.2754 MB |
| 5 | **97.75%** | **97.75%** | 0.0054s | 0.0039s | 0.0010s | 1.2752 MB |

## Observations

- Fixed classifier-only CKKS preserved FedAvg utility.
- Accuracy was comparable to the plaintext FedAvg baseline.
- Encrypted upload remained around 1.275 MB per round.
- Cryptographic computation was effectively negligible.

---

# EXP-007B: FedDyn + Fixed Classifier-Only CKKS under Extreme Non-IID

## Objective

Evaluate whether CKKS overhead profiling can be integrated with FedDyn under extreme Non-IID conditions without degrading the FedDyn optimization path.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | Extreme Non-IID |
| Algorithm | FedDyn |
| Alpha | 0.005 |
| Seed | 7 |
| Encrypted Scope | Classifier layer only |
| CKKS Usage | Secure transmission and overhead profiling |

## Results

| Round | Accuracy | F1 Score | Avg Enc Time | HE Agg Time | Dec Time | Crypto Overhead | Encrypted Upload |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.00% | 33.33% | 0.0071s | 0.0070s | 0.0020s | 0.0593% | 1.2756 MB |
| 2 | 50.00% | 33.33% | 0.0058s | 0.0050s | 0.0010s | 0.0472% | 1.2745 MB |
| 3 | 76.25% | 74.95% | 0.0059s | 0.0040s | 0.0010s | 0.0451% | 1.2754 MB |
| 4 | 75.75% | 74.23% | 0.0060s | 0.0040s | 0.0010s | 0.0466% | 1.2757 MB |
| 5 | **90.75%** | **90.73%** | 0.0063s | 0.0050s | 0.0010s | 0.0507% | 1.2751 MB |

## Comparison

| Method | Accuracy | F1 Score |
|---|---:|---:|
| FedDyn seed 7, no CKKS | 90.75% | 90.73% |
| FedDyn + CKKS seed 7 | **90.75%** | **90.73%** |

## Observations

- CKKS did not degrade stable FedDyn performance.
- FedDyn was seed-sensitive, but with seed 7 the encrypted-profiled run matched the plaintext result.
- Crypto overhead stayed around 0.05%.

---

# EXP-007C: Multi-Krum + Fixed Classifier-Only CKKS under Byzantine Attack

## Objective

Evaluate whether Multi-Krum Byzantine resilience remains compatible with selective CKKS encryption.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | Moderate Non-IID |
| Aggregation | Multi-Krum |
| Attack | Sign-flip attack |
| Malicious Clients | 1 |
| Attack Scale | 5.0 |
| Encrypted Scope | Classifier layer only |

## Results

| Round | Accuracy | F1 Score | Selected Clients | Avg Enc Time | HE Agg Time | Crypto Overhead | Encrypted Upload |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 81.00% | 80.88% | [1, 2] | 0.0063s | 0.0040s | 0.0631% | 1.2751 MB |
| 2 | 83.00% | 82.63% | [2, 3] | 0.0060s | 0.0041s | 0.0614% | 1.2752 MB |
| 3 | 91.00% | 90.95% | [2, 3] | 0.0020s | 0.0030s | 0.0270% | 1.2751 MB |
| 4 | 93.50% | 93.48% | [2, 3] | 0.0059s | 0.0015s | 0.0570% | 1.2747 MB |
| 5 | **96.00%** | **96.00%** | [2, 3] | 0.0050s | 0.0053s | 0.0566% | 1.2745 MB |

## Observations

- Multi-Krum continued selecting non-malicious clients in most rounds.
- CKKS profiling did not interfere with Byzantine-resilient aggregation.
- Final performance remained high under sign-flip attack.
- The improvement over plaintext Multi-Krum is treated as stochastic training variation.

---

# EXP-007D: FedAvg + Adaptive Selective CKKS

## Objective

Move beyond fixed classifier encryption by selecting trainable tensors dynamically using update magnitude.

## Selection Strategy

For each client update:

```text
update_magnitude = ||client_state - global_state||
```

Trainable tensors were ranked by update magnitude. BatchNorm buffers such as `running_mean`, `running_var`, and `num_batches_tracked` were excluded.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | IID |
| Aggregation | FedAvg |
| Selection Strategy | Top-k trainable tensors by update norm |
| Top-K | 4 |
| Budget | No explicit privacy-focused budget |
| HE Scheme | CKKS |

## Results

| Round | Accuracy | F1 Score | Avg Enc Time | HE Agg Time | Crypto Overhead | Encrypted Upload |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 81.75% | 81.37% | 0.0057s | 0.0043s | 0.0562% | 1.2754 MB |
| 2 | 91.25% | 91.25% | 0.0070s | 0.0050s | 0.0666% | 1.2751 MB |
| 3 | 94.50% | 94.49% | 0.0052s | 0.0030s | 0.0495% | 1.2758 MB |
| 4 | 97.00% | 97.00% | 0.0064s | 0.0050s | 0.0624% | 1.2756 MB |
| 5 | **97.50%** | **97.50%** | 0.0060s | 0.0040s | 0.0557% | 1.2753 MB |

## Observations

- Adaptive selection preserved FedAvg utility.
- The method selected trainable tensors dynamically instead of hard-coding the classifier layer.
- Overhead stayed extremely low.
- This established adaptive CKKS as a viable extension, but it did not yet quantify privacy coverage.

---

# EXP-007E: Multi-Krum + Adaptive Selective CKKS under Byzantine Attack

## Objective

Evaluate adaptive selective CKKS with Multi-Krum under Byzantine sign-flip attack.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | Moderate Non-IID |
| Aggregation | Multi-Krum |
| Attack | Sign-flip attack |
| Malicious Clients | 1 |
| Top-K | 4 |
| HE Scheme | CKKS |

## Results

| Round | Accuracy | F1 Score | Selected Clients | Avg Enc Time | HE Agg Time | Crypto Overhead | Encrypted Upload |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 77.25% | 76.87% | [1, 2] | 0.0058s | 0.0044s | 0.0579% | 1.2757 MB |
| 2 | 87.50% | 87.50% | [1, 2] | 0.0054s | 0.0040s | 0.0584% | 1.2754 MB |
| 3 | 86.00% | 85.72% | [2, 3] | 0.0056s | 0.0045s | 0.0582% | 1.2752 MB |
| 4 | 93.50% | 93.47% | [2, 3] | 0.0055s | 0.0045s | 0.0560% | 1.2755 MB |
| 5 | **95.75%** | **95.74%** | [2, 3] | 0.0056s | 0.0044s | 0.0560% | 1.2754 MB |

## Observations

- Adaptive CKKS remained compatible with Byzantine-resilient Multi-Krum.
- The malicious client was excluded in later rounds.
- Accuracy remained high while cryptographic overhead stayed negligible.
- This run established robustness compatibility but did not yet emphasize privacy coverage.

---

# EXP-007F: Privacy-Focused Budgeted Adaptive CKKS with Multi-Krum

## Objective

Evaluate a privacy-focused budgeted adaptive CKKS strategy that encrypts a larger, more meaningful subset of trainable parameters while still remaining practical.

This experiment was introduced because the earlier Top-K = 4 adaptive runs were extremely lightweight but had low update coverage. EXP-007F increases the adaptive selection budget to better quantify privacy coverage and residual leakage risk.

## Setup

| Parameter | Value |
|---|---|
| Dataset Split | Moderate Non-IID |
| Aggregation | Multi-Krum |
| Attack | Sign-flip attack |
| Malicious Clients | 1 |
| Selection Strategy | Budgeted top-k trainable tensors by raw update norm |
| Top-K | 50 |
| Max Selected Plaintext Budget | 1,000,000 bytes |
| HE Scheme | CKKS |
| HE Library | TenSEAL |
| CKKS Usage | Privacy-focused adaptive encrypted transmission and overhead profiling |

## Results

| Round | Accuracy | F1 Score | Selected Clients | UCR / ICR | PER | AEQ | ILR / RRS | Enc Upload | Crypto Overhead |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 79.25% | 78.98% | [2, 3] | 0.1863 | 0.062340 | 2.99 | 0.8137 | 79.0611 MB | 3.5003% |
| 2 | 87.25% | 87.13% | [2, 3] | 0.1839 | 0.062340 | 2.95 | 0.8161 | 79.0615 MB | 3.4768% |
| 3 | 91.00% | 90.93% | [2, 3] | 0.1831 | 0.062340 | 2.94 | 0.8169 | 79.0581 MB | 3.4355% |
| 4 | 91.50% | 91.45% | [2, 3] | 0.1822 | 0.062340 | 2.92 | 0.8178 | 79.0648 MB | 3.4380% |
| 5 | **94.75%** | **94.75%** | [1, 2] | 0.1827 | 0.062340 | 2.93 | 0.8173 | 79.0634 MB | 2.6818% |

## Detailed Round-5 Cryptographic Metrics

| Metric | Value |
|---|---:|
| Plain selected update per client | 999,968 bytes |
| Encrypted selected update per client | ~20.72 MB |
| Total encrypted upload per round | 79.0634 MB |
| Average encryption time per client | 0.2698 s |
| HE aggregation time | 0.1559 s |
| Decryption time | 0.0380 s |
| Crypto overhead | 2.6818% |
| Parameter Encryption Ratio | 6.2340% |
| Update Coverage Ratio / Information Coverage Ratio | 18.27% |
| Residual Leakage Risk / Reconstruction Risk Score | 81.73% |
| Adaptive Encryption Quality | 2.93 |

## Comparison: Lightweight vs Privacy-Focused Adaptive CKKS

| Method | Budget | Accuracy | F1 Score | UCR / ICR | PER | AEQ | ILR / RRS | Enc Upload | Crypto Overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Budgeted Adaptive FedAvg | 20 KB | 97.00% | 97.00% | 0.0051 | 0.000379 | 13.58 | 0.9949 | 1.2751 MB | 0.0615% |
| Budgeted Adaptive Multi-Krum | 20 KB | 91.75% | 91.71% | 0.0103 | 0.000044 | 233.63 | 0.9897 | 1.2755 MB | 0.0320% |
| **Privacy-Focused Adaptive Multi-Krum** | **1 MB** | **94.75%** | **94.75%** | **0.1827** | **0.062340** | **2.93** | **0.8173** | **79.0634 MB** | **2.6818%** |

## Observations

- Increasing the budget from 20 KB to 1 MB significantly improved privacy coverage.
- UCR / ICR increased from approximately 1% to approximately 18%.
- PER increased to 6.234%, meaning a meaningful fraction of trainable parameters was encrypted.
- Residual Leakage Risk decreased from approximately 99% to approximately 82%.
- The higher privacy budget increased encrypted upload to approximately 79 MB per round.
- Despite the larger encryption budget, crypto overhead remained below 3.6% and final accuracy remained 94.75%.
- This demonstrates a practical privacy–overhead trade-off.

## Key Interpretation

EXP-007F shows that adaptive CKKS can be tuned based on deployment needs:

- **20 KB budget:** ultra-lightweight encryption with negligible overhead.
- **1 MB budget:** stronger privacy coverage with manageable overhead.
- **Full-model encryption:** maximum protection, but likely impractical for EfficientNet-scale models.

This supports the proposed framework as a configurable secure FL design rather than a fixed encryption pipeline.

---

# EXP-007 Master Summary

| Experiment | Setting | Method | Accuracy | F1 Score | Enc Upload | Crypto Overhead | Key Result |
|---|---|---|---:|---:|---:|---:|---|
| EXP-007A | IID | FedAvg + Fixed CKKS | 97.75% | 97.75% | 1.275 MB | ~0.05–0.06% | Utility preserved |
| EXP-007B | Extreme Non-IID | FedDyn + Fixed CKKS | 90.75% | 90.73% | 1.275 MB | ~0.05% | FedDyn preserved under CKKS |
| EXP-007C | Moderate Non-IID + Byzantine | Multi-Krum + Fixed CKKS | 96.00% | 96.00% | 1.275 MB | ~0.03–0.06% | Byzantine robustness preserved |
| EXP-007D | IID | FedAvg + Adaptive CKKS | 97.50% | 97.50% | 1.275 MB | ~0.05–0.07% | Adaptive selection preserved utility |
| EXP-007E | Moderate Non-IID + Byzantine | Multi-Krum + Adaptive CKKS | 95.75% | 95.74% | 1.275 MB | ~0.056% | Adaptive CKKS compatible with Multi-Krum |
| **EXP-007F** | Moderate Non-IID + Byzantine | **Privacy-Focused Budgeted Adaptive CKKS** | **94.75%** | **94.75%** | **79.063 MB** | **2.68–3.50%** | **Stronger privacy coverage with manageable overhead** |

---

# Main Findings

1. Selective CKKS can preserve model utility while protecting selected model tensors.
2. Fixed classifier-only CKKS is extremely lightweight but protects only a narrow part of the model.
3. Adaptive CKKS is more flexible because it selects tensors based on update magnitude.
4. Budgeted adaptive CKKS allows direct control over the privacy–overhead trade-off.
5. The 1 MB privacy-focused budget encrypted 6.234% of parameters and captured approximately 18.27% of update movement.
6. Multi-Krum and CKKS can be combined successfully under Byzantine attack.
7. CKKS does not improve accuracy directly; it protects selected updates while preserving learning.
8. Compared to the 2025 paper, EXP-007 provides deeper cryptographic profiling and introduces adaptive encryption quality metrics.

---

# Research Contribution from EXP-007

EXP-007 introduces a selective and adaptive CKKS-based encrypted aggregation strategy for modern federated medical image classification. Unlike the 2025 paper, which applies homomorphic encryption broadly without adaptive selection analysis, this work evaluates fixed, adaptive, and budgeted adaptive encryption policies.

The strongest contribution is the privacy-focused budgeted adaptive CKKS experiment, which shows that the encrypted subset can be increased from an ultra-lightweight setting to a more privacy-oriented setting while maintaining high model utility and manageable overhead.

This makes the proposed framework configurable: hospitals or consortium administrators can select an encryption budget depending on whether the priority is low latency, high privacy coverage, or a balanced trade-off.
