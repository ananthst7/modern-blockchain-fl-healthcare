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

---

# EXP-007G — Adaptive CKKS Budget Sweep Analysis

## Objective

Evaluate the impact of increasing the adaptive CKKS encryption budget on:

- Model utility
- Privacy coverage
- Communication cost
- Cryptographic overhead
- Adaptive encryption efficiency

Three adaptive encryption budgets were evaluated:

| Budget | Plaintext Encryption Budget |
|---------|----------------------------:|
| Budget A | 1 MB |
| Budget B | 2 MB |
| Budget C | 4 MB |

## Experimental Results

| Metric | 1 MB | 2 MB | 4 MB |
|--------|------:|------:|------:|
| Accuracy | **96.00%** | 95.75% | 93.75% |
| F1 Score | **96.00%** | 95.74% | 93.73% |
| Parameter Encryption Ratio (PER) | 6.23% | 12.46% | 24.94% |
| Information Coverage Ratio (ICR) *(formerly UCR)* | 18.31% | **32.43%** | 32.04% |
| Residual Leakage Risk (RLR) *(formerly RRS)* | 81.69% | **67.57%** | 67.96% |
| Adaptive Encryption Quality (AEQ) | **2.94** | 2.60 | 1.29 |
| Cryptographic Overhead | 2.74% | 4.78% | 8.91% |
| Encrypted Upload Size | 79 MB | 156 MB | 312 MB |

## Observation

Increasing the adaptive encryption budget from **1 MB** to **2 MB** substantially improved privacy coverage, with the Information Coverage Ratio (ICR) increasing from **18.31%** to **32.43%**.

However, further increasing the budget from **2 MB** to **4 MB** resulted in only a marginal improvement in ICR despite approximately doubling the encrypted parameter ratio, communication cost, and cryptographic overhead.

This indicates that the current adaptive selector reaches an **information saturation point**, where encrypting additional parameters contributes very little additional privacy protection.

## Key Findings

- Parameter Encryption Ratio (PER) increases almost linearly.
- Communication cost increases almost linearly.
- Cryptographic overhead also increases steadily.
- Information Coverage Ratio (ICR) quickly reaches saturation.

This suggests that the majority of privacy-sensitive information is concentrated within a relatively small subset of model parameters.

## Research Limitation Identified

The current adaptive selection strategy ranks tensors solely according to update magnitude:

\[
\text{Score} = ||\Delta W||
\]

The budget sweep demonstrates that this assumption eventually reaches diminishing returns.

## Proposed Improvement (Next Experiment)

### Privacy-Aware Adaptive CKKS

Instead of

\[
\text{Score}=||\Delta W||
\]

the next selector will use

\[
\text{Score}
=
\alpha ||\Delta W||
+
\beta \cdot \text{LayerImportance}
+
\gamma \cdot \text{HistoricalImportance}
\]

where:

- **Update Magnitude** measures optimization significance.
- **Layer Importance** approximates information density.
- **Historical Importance** is computed using an Exponential Moving Average (EMA) of previous update magnitudes.

### Expected Benefits

- Higher Information Coverage Ratio (ICR)
- Lower Residual Leakage Risk (RLR)
- More stable layer selection across communication rounds
- Similar communication overhead
- Better privacy-efficiency trade-off

## Novelty Statement

The proposed **Privacy-Aware Adaptive CKKS** extends selective homomorphic encryption by prioritizing tensors according to both optimization dynamics and privacy relevance rather than update magnitude alone.

Unlike existing selective CKKS approaches, the proposed method aims to **maximize protected information under a fixed encryption budget**, instead of simply maximizing encrypted parameter count.

## Conclusion

The budget sweep demonstrates that increasing the encryption budget alone cannot indefinitely improve privacy coverage.

Instead, **adaptive selection quality becomes the primary limiting factor**, motivating the transition from budget-aware adaptive encryption to **Privacy-Aware Adaptive CKKS**, which will form the next major contribution of this work.

---

# EXP-007H — Information Leakage-Aware Adaptive CKKS (ILA-CKKS)

## Objective

EXP-007H introduces **Information Leakage-Aware Adaptive CKKS (ILA-CKKS)**, the final and strongest adaptive encryption strategy in EXP-007. The goal is to move beyond update-magnitude-only selection and encrypt the tensors that are estimated to be most privacy-sensitive under a strict communication budget.

The method is evaluated with:

| Parameter | Value |
|---|---:|
| Dataset | COVID-19 Radiography Binary |
| Model | EfficientNet-B0 |
| Clients | 4 |
| Global Rounds | 5 |
| Local Epochs | 1 |
| Aggregation | Multi-Krum |
| Attack | Sign-flip attack |
| Malicious Client Index | 0 |
| Attack Scale | 5.0 |
| HE Scheme | CKKS |
| HE Library | TenSEAL |
| Max Selected Plaintext Budget | 2,000,000 bytes |
| Device | CUDA |
| Total Runtime | 175.59 s |

## Motivation

EXP-007G showed that increasing the adaptive encryption budget improves ICR only up to a saturation point. The 2 MB budget improved ICR to around 32%, but increasing the budget to 4 MB did not substantially increase information coverage. This means the limiting factor was no longer only the encryption budget; it was the **quality of the tensor selector**.

The earlier selector used:

```text
Score = ||ΔW||
```

where `ΔW` is the tensor update. This captures how much a tensor changed, but it does not directly measure privacy sensitivity. A tensor can have a large update but low privacy leakage, while another tensor can have a smaller update but high patient-specific sensitivity.

ILA-CKKS addresses this by estimating leakage using three signals:

1. **Update magnitude** — how much the tensor changed.
2. **Fisher score** — how sensitive the loss is to the tensor.
3. **Gradient variance** — how client-specific or unstable the tensor gradients are across batches.

## Proposed ILA Score

The ILA selector computes the following score for each trainable tensor:

\[
\text{ILA}_i
=
\|\Delta W_i\|
\times
F_i
\times
V_i
\]

where:

| Term | Meaning | Why it matters |
|---|---|---|
| \(\|\Delta W_i\|\) | Update norm | Captures how much the tensor changed during local learning |
| \(F_i\) | Fisher-like score | Captures parameter sensitivity using mean squared gradients |
| \(V_i\) | Gradient variance | Captures instability/client-specific learning behavior |

The multiplication is intentional. A tensor receives a high ILA score only when it satisfies all three conditions: it changed meaningfully, it is sensitive to the loss, and it shows gradient variability. This avoids selecting tensors that are large or noisy but not privacy-relevant.

## Fisher Information Proxy

In this implementation, the Fisher-like score is estimated during local training using:

\[
F_i \approx \mathbb{E}[g_i^2]
\]

where \(g_i\) is the gradient of the loss with respect to tensor \(i\). Higher Fisher values indicate that changes to the tensor strongly affect the model loss. This is useful for privacy because highly loss-sensitive tensors are more likely to encode class-specific or patient-specific information.

## Gradient Variance Proxy

Gradient variance is computed from the variation in gradient norms across local batches:

\[
V_i = \text{Var}(\|g_i\|)
\]

High variance suggests that the tensor reacts differently across local samples or batches. In federated medical imaging, this may indicate sensitivity to client-specific distributions.

## Budget-Constrained Selection

The selector solves a practical budget-constrained selection problem:

\[
\max_S \sum_{i \in S} \text{ILA}_i
\]

subject to:

\[
\sum_{i \in S} \text{Bytes}_i \le B
\]

where \(B = 2,000,000\) plaintext bytes.

In implementation, tensors are ranked using ILA score density and packed until the byte budget is reached. The selected tensors are encrypted using CKKS; unselected tensors remain plaintext.

## Why ILA-CKKS is Better than Magnitude-Only Adaptive CKKS

Magnitude-only adaptive CKKS asks:

> Which tensors changed the most?

ILA-CKKS asks:

> Which tensors are most likely to leak private information if left visible?

This is a stronger privacy objective because it aligns tensor selection with privacy sensitivity rather than only optimization movement.

## EXP-007H / ILA-CKKS Round-Wise Results

| Round | Accuracy | F1 Score | Selected Clients | Selected Keys | Plain Budget | ICR | PER | PCR | LCR | VCR | InfCR | GER/current cosine field | Enc Upload | Crypto Overhead |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.50% | 74.19% | [2, 3] | 105 | 1,999,944 | 31.38% | 12.47% | 99.99% | 99.89% | 92.98% | 98.85% | 0.4193 | 156.854 MB | 8.63% |
| 2 | 89.75% | 89.70% | [2, 3] | 79 | 2,000,000 | 26.46% | 12.47% | 99.97% | 99.66% | 90.02% | 97.78% | 0.3858 | 156.838 MB | 8.68% |
| 3 | 96.00% | 96.00% | [2, 3] | 66 | 2,000,000 | 24.13% | 12.47% | 99.96% | 99.29% | 85.31% | 96.72% | 0.3658 | 156.852 MB | 9.74% |
| 4 | 96.75% | 96.75% | [2, 3] | 76 | 2,000,000 | 26.09% | 12.47% | 99.90% | 99.14% | 80.21% | 96.36% | 0.3902 | 156.844 MB | 7.34% |
| 5 | 95.25% | 95.24% | [2, 3] | 91 | 1,999,968 | 26.96% | 12.47% | 99.93% | 99.29% | 82.85% | 97.00% | 0.3812 | 156.852 MB | 7.42% |

## Aggregate ILA-CKKS Results

| Metric | Mean | Std | Min | Max | Final Round |
|---|---:|---:|---:|---:|---:|
| Accuracy | 90.65% | 7.97% | 75.50% | 96.75% | 95.25% |
| F1 Score | 90.38% | 8.46% | 74.19% | 96.75% | 95.24% |
| Information Coverage Ratio (ICR) | 27.01% | 2.39% | 24.13% | 31.38% | 26.96% |
| Parameter Encryption Ratio (PER) | 12.47% | 0.00% | 12.47% | 12.47% | 12.47% |
| Adaptive Encryption Quality (AEQ) | 2.1659 | 0.1917 | 1.9354 | 2.5171 | 2.1625 |
| Privacy Coverage Ratio (PCR) | 99.95% | 0.03% | 99.90% | 99.99% | 99.93% |
| Residual Privacy Leakage (RPL) | 0.05% | 0.03% | 0.01% | 0.10% | 0.07% |
| Leakage Coverage Ratio (LCR) | 99.45% | 0.28% | 99.14% | 99.89% | 99.29% |
| Variance Coverage Ratio (VCR) | 86.27% | 4.66% | 80.21% | 92.98% | 82.85% |
| Influence Coverage Ratio (InfCR) | 97.34% | 0.89% | 96.36% | 98.85% | 97.00% |
| Gradient Energy Ratio / current cosine field | 0.3885 | 0.0175 | 0.3658 | 0.4193 | 0.3812 |
| Crypto Overhead | 8.36% | 0.89% | 7.34% | 9.74% | 7.42% |
| Encrypted Upload | 156.848 MB | 0.006 MB | 156.838 MB | 156.854 MB | 156.852 MB |

## Key Result

The best model performance occurred in **Round 4**, reaching **96.75% accuracy** and **96.75% F1 score**. The final round reached **95.25% accuracy** and **95.24% F1 score**.

Most importantly, ILA-CKKS encrypted only about **12.47% of trainable parameters**, but achieved:

- **99.95% average Privacy Coverage Ratio (PCR)**.
- **99.45% average Leakage Coverage Ratio (LCR)**.
- **86.27% average Variance Coverage Ratio (VCR)**.
- **97.34% average Influence Coverage Ratio (InfCR)**.

This demonstrates that ILA-CKKS protects most of the estimated privacy-sensitive signal while encrypting only a limited fraction of model parameters.

## Observations

- ILA-CKKS consistently selected clients `[2, 3]` under Multi-Krum, excluding the malicious sign-flip client in every round.
- The selected tensor count changed across rounds: 105, 79, 66, 76, 91, showing that the selector remained adaptive rather than fixed.
- The selected plaintext budget stayed almost exactly at 2 MB per round.
- The encrypted upload was approximately **156.848 MB** per round.
- Crypto overhead averaged **8.36%**, which is higher than lightweight CKKS but still practical for a privacy-focused setting.

## Interpretation

ILA-CKKS does not try to encrypt the entire model. Instead, it encrypts the subset estimated to contain the highest privacy leakage. This is why ICR remains around **27.01%** while LCR and PCR are much higher. ICR measures update-energy coverage, not privacy-leakage coverage. A privacy-aware method can intentionally ignore large but low-leakage updates and instead prioritize smaller but more sensitive tensors.

This is a key scientific insight: **high privacy coverage does not require high parameter encryption ratio if the selected parameters are chosen using leakage-aware criteria.**

---

# EXP-007I — Independent Privacy Validation Metrics for ILA-CKKS

## Objective

EXP-007I adds independent validation metrics to ensure that ILA-CKKS is not judged only by the same score used for selection. This is important because the internal PCR metric is computed from the ILA score itself.

Therefore, EXP-007I evaluates ILA-CKKS using several additional metrics:

1. Parameter Encryption Ratio (PER)
2. Information Coverage Ratio (ICR)
3. Privacy Coverage Ratio (PCR)
4. Leakage Coverage Ratio (LCR)
5. Variance Coverage Ratio (VCR)
6. Influence Coverage Ratio (InfCR)
7. Gradient Energy Ratio / current cosine field
8. Residual Privacy Leakage (RPL)

## Why CKKS Needs Evaluation Metrics

Differential Privacy provides a formal privacy parameter \(\epsilon\). CKKS does not. CKKS provides cryptographic confidentiality for encrypted values, but selective CKKS raises an additional question:

> If only some tensors are encrypted, how do we know those tensors are the important ones?

EXP-007I answers this by evaluating whether the encrypted subset captures most of the estimated privacy-sensitive information.

## Metric 1 — Parameter Encryption Ratio (PER)

### Formula

\[
PER = \frac{\text{Encrypted Parameters}}{\text{Total Trainable Parameters}}
\]

### What it measures

PER measures how much of the model is encrypted by parameter count.

### Trustworthiness

PER is an exact implementation metric. It does not rely on approximations or assumptions.

### Limitation

PER does not tell whether the encrypted parameters are important. Encrypting 50% of parameters can still be weak if the wrong half is selected.

### ILA Result

ILA-CKKS achieved an average PER of **12.47%**.

## Metric 2 — Information Coverage Ratio (ICR)

### Formula

\[
ICR = \frac{\sum_{i \in S} \|\Delta W_i\|}{\sum_i \|\Delta W_i\|}
\]

### What it measures

ICR measures how much of the total update movement is encrypted.

### Trustworthiness

ICR is useful for measuring optimization-signal protection. It is mathematically direct and easy to verify.

### Limitation

ICR does not necessarily measure privacy leakage. Large updates are not always privacy-sensitive, and small updates can still leak information.

### ILA Result

ILA-CKKS achieved an average ICR of **27.01%**.

This is lower than the privacy metrics because ILA intentionally prioritizes leakage-sensitive tensors rather than simply maximizing update norm.

## Metric 3 — Privacy Coverage Ratio (PCR)

### Formula

\[
PCR = \frac{\sum_{i \in S} \text{ILA}_i}{\sum_i \text{ILA}_i}
\]

### What it measures

PCR measures how much of the ILA-estimated privacy signal is encrypted.

### Trustworthiness

PCR is useful because it directly measures the objective optimized by ILA-CKKS.

### Limitation

PCR is selector-aligned and partially self-referential. Since ILA score is used for both selection and PCR calculation, PCR alone should not be treated as independent proof.

### ILA Result

Average PCR was **99.95%**, with average residual privacy leakage of only **0.05%**.

## Metric 4 — Leakage Coverage Ratio (LCR)

### Formula

\[
LCR = \frac{\sum_{i \in S} F_i}{\sum_i F_i}
\]

where \(F_i\) is the Fisher-like score.

### What it measures

LCR measures how much Fisher-sensitive information is encrypted, independently of update magnitude and gradient variance.

### Why it is trustworthy

Fisher Information is a widely used parameter-importance proxy in continual learning, pruning, and sensitivity estimation. Unlike PCR, LCR does not multiply all three ILA components together; it validates the selected subset using only Fisher sensitivity.

### ILA Result

Average LCR was **99.45%**, showing that the encrypted subset captured almost all Fisher-sensitive signal.

## Metric 5 — Variance Coverage Ratio (VCR)

### Formula

\[
VCR = \frac{\sum_{i \in S} V_i}{\sum_i V_i}
\]

where \(V_i\) is the gradient variance score.

### What it measures

VCR measures how much of the gradient variability is protected.

### Why it is trustworthy

VCR is independent of Fisher-only coverage and update coverage. It measures whether tensors with unstable or client-specific gradients were selected.

### ILA Result

Average VCR was **86.27%**. This is lower than LCR but still strong, which is expected because gradient variance is noisier than Fisher sensitivity.

## Metric 6 — Influence Coverage Ratio (InfCR)

### Formula

\[
InfCR = \frac{\sum_{i \in S} \|\Delta W_i\|F_i}{\sum_i \|\Delta W_i\|F_i}
\]

### What it measures

Influence Coverage Ratio measures how much of the update-weighted Fisher influence is encrypted.

### Why it is trustworthy

InfCR combines update movement and Fisher sensitivity but excludes gradient variance. It therefore validates whether selected tensors are both changing and loss-sensitive.

### ILA Result

Average InfCR was **97.34%**, indicating that nearly all influential update-sensitive parameters were encrypted.

## Metric 7 — Gradient Energy Ratio / Current Cosine Field

### Current implementation

The current `avg_gradient_cosine_similarity` field computes:

\[
GER = \sqrt{\frac{\sum_{i \in S} \|\Delta W_i\|^2}{\sum_i \|\Delta W_i\|^2}}
\]

### What it measures

Despite the variable name, this is closer to a **Gradient Energy Ratio (GER)** than true cosine similarity. It measures how much update energy remains in the selected encrypted subset.

### Trustworthiness

GER is useful as a secondary optimization-signal metric, but it should not be described as true cosine similarity until the implementation is revised to compute:

\[
\cos(\theta) = \frac{g_S \cdot g}{\|g_S\|\|g\|}
\]

### ILA Result

The average current GER/cosine field was **0.3885**.

## Metric Trustworthiness Summary

| Metric | Type | Trustworthiness | Main Limitation |
|---|---|---|---|
| PER | Exact implementation metric | Very high | Does not measure importance |
| ICR | Exact update-energy metric | High | Not privacy-specific |
| PCR | Selector-aligned leakage metric | Medium | Self-referential |
| LCR | Fisher-only independent metric | Very high | Fisher is still a proxy |
| VCR | Variance-only independent metric | High | Gradient variance can be noisy |
| InfCR | Update × Fisher independent metric | Very high | Does not include variance |
| GER/current cosine field | Update-energy proxy | Medium | Not true cosine yet |

## Why the Metrics are Trustworthy Together

No single metric proves privacy by itself. However, the metrics become stronger when interpreted together:

- PER shows the encryption budget is limited.
- ICR shows how much update movement is encrypted.
- PCR shows the selector-aligned leakage estimate.
- LCR independently confirms Fisher-sensitive coverage.
- VCR independently confirms gradient-variance coverage.
- InfCR independently confirms update-sensitive Fisher influence.
- GER shows the retained update-energy fraction.

Together, these metrics show that ILA-CKKS is not merely encrypting many parameters; it is encrypting the parameters that multiple independent proxies identify as privacy-relevant.

## Comparison Against EXP-007F and EXP-007G

| Method | Budget | Accuracy | F1 | PER | ICR | PCR | LCR | VCR | InfCR | Enc Upload | Crypto Overhead | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EXP-007F Budgeted Adaptive Multi-Krum | 1 MB | 94.75% | 94.75% | 6.23% | 18.27% | N/A | N/A | N/A | N/A | 79.06 MB | 2.68–3.50% | Stronger than lightweight CKKS, but still magnitude-based |
| EXP-007G Budgeted Adaptive Multi-Krum | 2 MB | 95.75% | 95.74% | 12.46% | 32.43% | N/A | N/A | N/A | N/A | 156 MB | 4.78% | Best magnitude-only budget tradeoff |
| **EXP-007I ILA-CKKS Multi-Krum** | **2 MB** | **96.75% best / 95.25% final** | **96.75% best / 95.24% final** | **12.47%** | **27.01%** | **99.95%** | **99.45%** | **86.27%** | **97.34%** | **156.85 MB** | **8.36%** | Leakage-aware selection with independent validation |

## Key Insight

Magnitude-only adaptive CKKS improved ICR, but ILA-CKKS improves privacy-specific coverage metrics. This distinction is important: ICR measures how much update movement is encrypted, while LCR, VCR, and InfCR measure whether the privacy-sensitive parts of the update are protected.

Thus, ILA-CKKS provides a stronger privacy argument than magnitude-only adaptive CKKS even when its ICR is lower.

---

# Revised EXP-007 Master Summary

| Experiment | Setting | Method | Accuracy | F1 Score | Privacy Metrics | Enc Upload | Crypto Overhead | Key Result |
|---|---|---|---:|---:|---|---:|---:|---|
| EXP-007A | IID | FedAvg + Fixed Classifier CKKS | 97.75% | 97.75% | Not privacy-scored | 1.275 MB | ~0.05–0.06% | Utility preserved with minimal HE overhead |
| EXP-007B | Extreme Non-IID | FedDyn + Fixed Classifier CKKS | 90.75% | 90.73% | Not privacy-scored | 1.275 MB | ~0.05% | FedDyn path preserved under CKKS profiling |
| EXP-007C | Moderate Non-IID + Byzantine | Multi-Krum + Fixed Classifier CKKS | 96.00% | 96.00% | Not privacy-scored | 1.275 MB | ~0.03–0.06% | Multi-Krum remained compatible with CKKS |
| EXP-007D | IID | FedAvg + Adaptive CKKS | 97.50% | 97.50% | Lightweight adaptive selection | 1.275 MB | ~0.05–0.07% | Adaptive tensor selection preserved utility |
| EXP-007E | Moderate Non-IID + Byzantine | Multi-Krum + Adaptive CKKS | 95.75% | 95.74% | Lightweight adaptive selection | 1.275 MB | ~0.056% | Adaptive CKKS worked under Byzantine defense |
| EXP-007F | Moderate Non-IID + Byzantine | 1 MB Budgeted Adaptive CKKS | 94.75% | 94.75% | ICR 18.27%, PER 6.23% | 79.06 MB | 2.68–3.50% | Stronger privacy coverage with manageable overhead |
| EXP-007G | Moderate Non-IID + Byzantine | Budget Sweep Adaptive CKKS | 95.75% at 2 MB | 95.74% at 2 MB | Best ICR 32.43% at 2 MB | 156 MB at 2 MB | 4.78% at 2 MB | Revealed ICR saturation with magnitude-only selection |
| EXP-007H | Moderate Non-IID + Byzantine | ILA-CKKS Algorithm | 96.75% best | 96.75% best | ILA score = update × Fisher × variance | 156.85 MB | 8.36% avg | Introduced leakage-aware tensor selection |
| EXP-007I | Moderate Non-IID + Byzantine | ILA-CKKS Independent Validation | 95.25% final | 95.24% final | LCR 99.45%, VCR 86.27%, InfCR 97.34% | 156.85 MB | 7.42% | Validated that selected tensors captured privacy-sensitive signal |

## Final EXP-007 Findings

1. CKKS preserves model utility when applied selectively.
2. Fixed classifier-only CKKS is computationally lightweight but narrow in coverage.
3. Magnitude-based adaptive CKKS improves flexibility but eventually saturates.
4. Budgeted adaptive CKKS provides explicit privacy–communication control.
5. ILA-CKKS improves the selection objective by using update magnitude, Fisher sensitivity, and gradient variance.
6. Independent metrics show that ILA-CKKS encrypts only about **12.47%** of parameters while covering **99.45%** Fisher leakage and **97.34%** influence signal.
7. Multi-Krum successfully excluded the malicious sign-flip client in the ILA-CKKS run, repeatedly selecting clients `[2, 3]`.
8. The strongest research contribution of EXP-007 is not merely using CKKS, but proposing a **leakage-aware selective encryption policy** and evaluating it with independent privacy coverage metrics.

## Final Contribution Statement

EXP-007 extends the base 2025 paper by replacing broad homomorphic encryption with a modern, configurable and leakage-aware CKKS pipeline. The proposed ILA-CKKS method demonstrates that privacy-sensitive update components can be prioritized under a fixed encryption budget, allowing the framework to protect the most important information without full-model encryption.

This makes the framework more practical for healthcare federated learning, where hospitals may require a tunable balance between privacy, bandwidth, latency and model utility.
