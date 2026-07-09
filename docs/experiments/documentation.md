# Modern Blockchain-Enabled Federated Learning with Homomorphic Encryption for Healthcare

## 1. Project Overview

This repository implements a modern privacy-preserving federated learning framework for healthcare image classification.

It is based on and extends the 2025 paper:

> **Blockchain-based Federated Learning with Homomorphic Encryption for Privacy-Preserving Healthcare Data Sharing**

The original paper proposed a healthcare FL system using:

- Federated Learning
- Homomorphic Encryption
- Blockchain
- CNN model
- COVID-19 and Brain Tumor datasets
- Ganache / Truffle blockchain environment

This project modernizes and extends that system using:

- **EfficientNet-B0** instead of the original CNN.
- **PyTorch** instead of TensorFlow/Keras.
- **FedAvg** and **FedDyn** federated optimization.
- **Multi-Krum** Byzantine-resilient aggregation.
- **CKKS homomorphic encryption** using TenSEAL.
- **Information Leakage-Aware Adaptive CKKS (ILA-CKKS)**.
- **Differential Privacy analysis** using both approximate update perturbation and formal Opacus DP-SGD.
- **Proof-of-Authority blockchain audit ledger**.
- **PoA vs PoW consensus comparison**.
- **Hyperledger Fabric-style smart contract abstraction**.

---

## 2. Research Goal

Healthcare institutions often cannot share raw patient data because medical data contains sensitive protected health information.

Federated Learning solves part of this problem by keeping data local:

> Hospitals train locally and share only model updates.

However, standard FL still has security and privacy issues:

- A central server can become a single point of failure.
- Model updates can leak information.
- Malicious hospitals can poison the global model.
- There may be no transparent audit trail.
- Contribution tracking is difficult.

This project builds a more complete healthcare FL framework with:

\[
\text{Federated Learning} + \text{Byzantine Defense} + \text{CKKS Encryption} + \text{Privacy Analysis} + \text{Blockchain Audit}
\]

---

## 3. Final High-Level Architecture

```text
Hospitals / Clients
        |
        | Local EfficientNet-B0 training
        v
Local Model Updates
        |
        | Optional classifier perturbation / DP-SGD
        v
Byzantine Filtering using Multi-Krum
        |
        | Selective CKKS Encryption using ILA-CKKS
        v
Secure Aggregation / Global Model Update
        |
        | Hashes, metrics, rewards, global model hash
        v
PoA Blockchain Audit Ledger
        |
        | Optional Fabric-style contract abstraction
        v
Auditable Healthcare FL System
```

---

## 4. Dataset

The project currently documents experiments on:

| Dataset | Task |
|---|---|
| COVID-19 Radiography Database | Binary COVID vs Normal classification |

Dataset preparation used:

- COVID class
- Normal class
- 4 simulated hospital clients
- IID, moderate Non-IID, and extreme Non-IID splits

### Missing Dataset Result

The base paper also used a Brain Tumor MRI dataset. Brain MRI validation is not present in the uploaded/current documented logs and should be treated as future work unless those runs are added later.

---

## 5. Core Definitions

### Federated Learning

Federated Learning trains a global model without moving raw data from hospitals.

For each round \(t\):

1. Server sends global model \(w_t\).
2. Hospital \(i\) trains locally on private data.
3. Hospital returns update \(w_i^t\) or \(\Delta w_i^t\).
4. Server aggregates updates.

### FedAvg

FedAvg computes a weighted average of local client models:

\[
w_{t+1} = \sum_{i=1}^{K} \frac{n_i}{N} w_i^t
\]

where:

- \(K\) = number of clients
- \(n_i\) = samples at client \(i\)
- \(N = \sum_i n_i\)
- \(w_i^t\) = locally trained model from client \(i\)

### Non-IID Data

Non-IID means hospitals do not have the same class distribution.

Example:

| Client | COVID | Normal |
|---|---:|---:|
| Client 1 | 390 | 10 |
| Client 2 | 390 | 10 |
| Client 3 | 10 | 390 |
| Client 4 | 10 | 390 |

This makes FL harder because each hospital learns biased local patterns.

### FedDyn

FedDyn modifies local training with dynamic regularization to reduce client drift under Non-IID data. It is useful when FedAvg struggles with heterogeneous hospital distributions.

### Byzantine Attack

A Byzantine client sends malicious model updates.

This project uses sign-flip attack:

\[
\Delta W_{malicious} = -s \cdot \Delta W
\]

where \(s = 5.0\).

### Multi-Krum

Multi-Krum is a Byzantine-resilient aggregation method. It selects updates that are closest to other updates and rejects outlier updates.

This helps prevent poisoned updates from damaging the global model.

---

## 6. Homomorphic Encryption

### What is Homomorphic Encryption?

Homomorphic Encryption allows computation on encrypted values.

For an encryption function \(Enc\):

\[
Dec(Enc(a) + Enc(b)) \approx a + b
\]

This means encrypted model updates can be aggregated without exposing plaintext values.

### CKKS

CKKS is an approximate homomorphic encryption scheme for real-valued vectors.

This project uses **TenSEAL CKKS** because neural network weights and updates are floating-point values.

### CKKS Configuration

| Parameter | Value |
|---|---|
| HE Scheme | CKKS |
| Library | TenSEAL |
| Polynomial Modulus Degree | 8192 |
| Coefficient Modulus Bits | [60, 40, 40, 60] |
| Global Scale | \(2^{40}\) |
| Model | EfficientNet-B0 |

---

## 7. Selective CKKS

Full-model encryption is expensive for EfficientNet-B0.

Therefore, this project evaluates selective encryption:

1. Fixed classifier-only CKKS
2. Adaptive CKKS by update magnitude
3. Budgeted adaptive CKKS
4. Information Leakage-Aware CKKS

### Why Selective Encryption?

EfficientNet-B0 has many parameters. Encrypting all updates can be expensive.

Selective CKKS tries to answer:

> Which parameters should be encrypted to protect the most important information under a limited communication budget?

---

## 8. ILA-CKKS: Main Encryption Contribution

### Problem with Magnitude-Only Selection

Earlier adaptive CKKS used:

\[
Score_i = \|\Delta W_i\|
\]

This selects tensors that changed the most, but large changes are not always the most privacy-sensitive.

### ILA Score

Information Leakage-Aware CKKS uses:

\[
ILA_i = \|\Delta W_i\| \times F_i \times V_i
\]

where:

| Term | Meaning |
|---|---|
| \(\|\Delta W_i\|\) | Update magnitude |
| \(F_i\) | Fisher-like sensitivity |
| \(V_i\) | Gradient variance |

### Fisher Proxy

\[
F_i \approx \mathbb{E}[g_i^2]
\]

where \(g_i\) is the gradient for parameter tensor \(i\).

High Fisher score means that parameter strongly affects the loss.

### Gradient Variance

\[
V_i = Var(\|g_i\|)
\]

High variance suggests client-specific or unstable gradients.

### Budget Constraint

ILA-CKKS selects tensors under a fixed plaintext byte budget:

\[
\max_S \sum_{i \in S} ILA_i
\]

subject to:

\[
\sum_{i \in S} Bytes_i \le B
\]

where \(B = 2,000,000\) bytes in the final ILA experiment.

---

---

# 9. CKKS Privacy Metrics

Unlike Differential Privacy, CKKS Homomorphic Encryption does **not** provide a single numerical privacy guarantee such as ε (epsilon).

Instead, this project evaluates **how much potentially sensitive information is protected by encryption** using a collection of proxy coverage metrics.

These metrics measure **what fraction of important model updates are encrypted**, rather than attempting to estimate cryptographic security.

---

## Why These Metrics?

Simply reporting the **percentage of encrypted parameters** is not sufficient.

For example:

- Encrypting 10% of parameters could protect almost all important information.
- Encrypting 50% of parameters could still miss the most privacy-sensitive tensors.

Therefore, this project evaluates encryption quality from multiple perspectives:

- Parameter coverage
- Update magnitude coverage
- Fisher Information coverage
- Gradient variance coverage
- Information Leakage-Aware (ILA) coverage

---

## Privacy Coverage Metrics

| Metric | Formula (Plain Text) | Meaning | Higher is Better? | Notes |
|---------|----------------------|---------|-------------------|-------|
| **PER** (Parameter Encryption Ratio) | Encrypted Parameters / Total Trainable Parameters | Percentage of model parameters encrypted | ✅ Yes | Does not consider parameter importance |
| **ICR** (Importance Coverage Ratio) | Sum of Selected Update Magnitudes / Sum of All Update Magnitudes | Fraction of update movement protected | ✅ Yes | Based only on update magnitude |
| **PCR** (Privacy Coverage Ratio) | Sum of Selected ILA Scores / Sum of All ILA Scores | Fraction of estimated privacy-sensitive information protected | ✅ Yes | Primary metric for ILA-CKKS |
| **RPL** (Residual Privacy Leakage) | 1 − PCR | Estimated remaining information not encrypted | ❌ Lower is Better | Proxy leakage estimate |
| **LCR** (Leakage Coverage Ratio) | Sum of Selected Fisher Scores / Sum of All Fisher Scores | Coverage of Fisher-sensitive parameters | ✅ Yes | Fisher Information is a proxy |
| **VCR** (Variance Coverage Ratio) | Sum of Selected Gradient Variances / Sum of All Gradient Variances | Coverage of unstable or client-specific gradients | ✅ Yes | High variance often indicates sensitive updates |
| **InfCR** (Influence Coverage Ratio) | Sum of (Selected Update Magnitude × Fisher) / Sum of (All Update Magnitude × Fisher) | Coverage of influential parameter updates | ✅ Yes | Combines update size with Fisher Information |
| **GER** (Gradient Energy Retention) | Square Root of (Selected Gradient Energy / Total Gradient Energy) | Amount of gradient energy preserved after selection | ✅ Yes | Indicates how much optimization signal is retained |

---

## Explanation of Each Metric

### 1. PER — Parameter Encryption Ratio

Measures **how much of the model is encrypted**.

Calculation:

```
PER = Encrypted Parameters / Total Trainable Parameters
```

Interpretation:

- Higher PER means more parameters are encrypted.
- Does **not** indicate whether the encrypted parameters are actually important.

---

### 2. ICR — Importance Coverage Ratio

Measures how much **model update magnitude** is protected.

Calculation:

```
ICR =
Sum of Selected Update Magnitudes
---------------------------------
Sum of All Update Magnitudes
```

Interpretation:

- Large model updates often contain important learning information.
- Higher ICR means more update movement is encrypted.

Limitation:

Magnitude alone does not necessarily correspond to privacy leakage.

---

### 3. PCR — Privacy Coverage Ratio

The primary metric introduced for **Information Leakage-Aware CKKS (ILA-CKKS)**.

Calculation:

```
PCR =
Sum of Selected ILA Scores
--------------------------
Sum of All ILA Scores
```

where

```
ILA Score =
Update Magnitude
× Fisher Information
× Gradient Variance
```

Interpretation:

- Higher PCR indicates that the encryption process protects a larger fraction of estimated privacy-sensitive information.
- This is the most important metric for evaluating ILA-CKKS.

---

### 4. RPL — Residual Privacy Leakage

Measures the estimated fraction of information **not protected**.

Calculation:

```
RPL = 1 − PCR
```

Interpretation:

- Lower values are better.
- An RPL of 0 means all estimated sensitive information is encrypted.

---

### 5. LCR — Leakage Coverage Ratio

Measures coverage of parameters with high Fisher Information.

Calculation:

```
LCR =
Sum of Selected Fisher Scores
-----------------------------
Sum of All Fisher Scores
```

Interpretation:

- Fisher Information estimates how strongly a parameter affects model predictions.
- Encrypting high-Fisher tensors generally provides better privacy protection.

---

### 6. VCR — Variance Coverage Ratio

Measures protection of parameters with large gradient variance.

Calculation:

```
VCR =
Sum of Selected Gradient Variances
----------------------------------
Sum of All Gradient Variances
```

Interpretation:

- Large gradient variance often indicates client-specific information.
- Encrypting these parameters reduces potential information leakage.

---

### 7. InfCR — Influence Coverage Ratio

Combines update magnitude with Fisher Information.

Calculation:

```
InfCR =
Sum of (Selected Update Magnitude × Fisher)
-------------------------------------------
Sum of (All Update Magnitude × Fisher)
```

Interpretation:

- Gives higher importance to parameters that both:
  - change significantly, and
  - strongly affect the model.

---

### 8. GER — Gradient Energy Retention

Measures how much optimization signal remains after parameter selection.

Calculation:

```
GER =
Square Root(
Selected Gradient Energy
------------------------
Total Gradient Energy
)
```

Interpretation:

- Higher GER means the selected encrypted tensors retain more learning information.
- Useful for understanding communication-efficiency versus optimization quality.

---

## Relationship Between the Metrics

```
Parameter Coverage
        │
        ▼
       PER
        │
        ▼
Importance Coverage
        │
        ├────────► ICR
        │
        ├────────► LCR
        │
        ├────────► VCR
        │
        └────────► InfCR
                     │
                     ▼
                 ILA Score
                     │
                     ▼
                    PCR
                     │
                     ▼
               RPL = 1 − PCR
```

---

## Important Note

These metrics **do not replace formal Differential Privacy**.

They estimate **how much potentially sensitive model information is encrypted**, but they do **not** provide mathematical privacy guarantees like ε-Differential Privacy.

Therefore:

- CKKS provides **cryptographic protection** for encrypted computations.
- Differential Privacy provides **mathematical privacy guarantees**.
- The metrics above evaluate the **effectiveness of selective encryption**, not formal privacy.

For this reason, they should be interpreted as **privacy coverage metrics** rather than **formal privacy guarantees**.

## 10. Differential Privacy Analysis

This project evaluates two privacy-noise approaches.

### 10.1 Classifier-Only Update Perturbation

The implemented update perturbation applies Gaussian noise only to the classifier layer.

This is useful for utility-preserving perturbation, but it is not whole-model DP.

Approximate Gaussian mechanism:

\[
\epsilon \approx \frac{S \sqrt{2\ln(1.25/\delta)}}{\sigma}
\]

where:

- \(S\) = sensitivity
- \(\sigma\) = noise standard deviation
- \(\delta = 10^{-5}\)
- replace-one update sensitivity \(S = 2C\)
- add/remove update sensitivity \(S = C\)

Original configuration:

| Parameter | Value |
|---|---:|
| Clip norm | 100 |
| Noise std | 1×10⁻⁶ |
| Replace-one ε | 968,961,052.52 |
| Add/remove ε | 484,480,526.26 |

Interpretation:

> This is utility-focused perturbation, not strong DP.

### 10.2 Opacus DP-SGD

Opacus DP-SGD provides formal epsilon accounting using per-sample gradient clipping and Gaussian noise.

Best formal utility result:

| Method | Epsilon | Accuracy | F1 |
|---|---:|---:|---:|
| Opacus DP-SGD | 173.04 | 84.75% | 84.70% |

Strongest formal privacy result:

| Method | Epsilon | Accuracy |
|---|---:|---:|
| Opacus DP-SGD | 1.76 | 57.00% |

Conclusion:

> Formal DP gives measurable privacy but causes substantial accuracy degradation.

---

## 11. Blockchain Layer

The base paper used Ganache/Truffle. This project adds a more explicit blockchain analysis.

### EXP-008: Proof-of-Authority Audit Ledger

A Python PoA blockchain audit ledger records:

- client update hashes
- encrypted update hashes
- aggregation hash
- global model hash
- metrics
- rewards
- Merkle root
- previous block hash

Results:

| Metric | Value |
|---|---:|
| Blocks including genesis | 6 |
| Training round blocks | 5 |
| Transactions | 31 |
| Average block creation time | 0.10874 ms |
| Verification time | 0.5587 ms |
| Ledger size | 28.37207 KB |
| Chain valid | true |
| Tamper detected | true |

No raw patient data or raw model weights are stored on-chain.

### EXP-009: PoA vs PoW

| Metric | PoA | PoW |
|---|---:|---:|
| Avg block creation time | 0.09064 ms | 116.01598 ms |
| Max block creation time | 0.1048 ms | 260.7508 ms |
| Verification time | 0.4323 ms | 0.4595 ms |
| Tamper detected | true | true |

PoW slowdown:

\[
\frac{116.01598}{0.09064} \approx 1279.96
\]

Conclusion:

> PoA is better suited for cross-silo healthcare FL because hospitals and validators are known permissioned entities.

### EXP-010: Fabric-Style Contract Abstraction

The project also implements a Hyperledger Fabric-style smart contract abstraction.

Functions:

- RegisterHospital
- SubmitUpdateHash
- SubmitAggregation
- IssueRewards
- QueryRound
- QueryHospitalHistory
- VerifyContractState

Results:

| Metric | Value |
|---|---:|
| Rounds | 5 |
| Hospitals | 4 |
| World state assets | 49 |
| History transactions | 49 |
| Avg register hospital time | 0.05685 ms |
| Avg submit update time | 0.04186 ms |
| Avg submit aggregation time | 0.05494 ms |
| Avg issue rewards time | 0.0954 ms |
| Avg query round time | 0.00426 ms |

Important limitation:

> This is a Fabric-style abstraction, not a deployed Hyperledger Fabric Docker network.

---

## 12. Final Results Summary

| Experiment | Method | Accuracy | F1 | Key Contribution |
|---|---|---:|---:|---|
| 2025 Base Paper | CNN + FedAvg + HE + Blockchain | 97.25% | N/A | Original baseline |
| EXP-001 | Centralized EfficientNet-B0 | 98.00% | 98.00% | Strong modern model |
| EXP-002 | FedAvg IID | 97.50% | 97.50% | Federated baseline |
| EXP-003 | FedAvg moderate Non-IID | 97.00% | 97.00% | Heterogeneity analysis |
| EXP-003B | FedAvg extreme Non-IID | 82.00% | 82.00% | FedAvg failure case |
| EXP-004A | FedDyn extreme Non-IID | 92.25% | 92.25% | Robust optimization |
| EXP-005E | Multi-Krum IID Byzantine | 97.00% | 97.00% | Byzantine defense |
| EXP-005F | Multi-Krum moderate Non-IID Byzantine | 93.50% | 93.47% | Realistic Byzantine defense |
| EXP-006C | Opacus DP-SGD | 84.75% | 84.70% | Formal DP utility |
| EXP-007I | ILA-CKKS | 95.25% final | 95.24% final | Leakage-aware encryption |
| EXP-008 | PoA blockchain | N/A | N/A | Auditability + tamper detection |
| EXP-009 | PoA vs PoW | N/A | N/A | Consensus suitability |
| EXP-010 | Fabric-style contract | N/A | N/A | Permissioned blockchain design |

---

## 13. Comparison with Base Paper

| Aspect | 2025 Base Paper | This Project |
|---|---|---|
| Model | CNN | EfficientNet-B0 |
| Framework | TensorFlow/Keras | PyTorch |
| FL Algorithm | FedAvg | FedAvg + FedDyn |
| Heterogeneity analysis | Limited | IID, moderate Non-IID, extreme Non-IID |
| Byzantine attacks | Not evaluated | Sign-flip attack |
| Robust aggregation | Not included | Multi-Krum |
| HE Library | Pyfhel / Microsoft SEAL | TenSEAL CKKS |
| HE Strategy | General HE | Fixed, adaptive, budgeted, ILA-CKKS |
| DP | Not evaluated | Approximate perturbation + formal Opacus DP-SGD |
| Blockchain | Ganache/Truffle | PoA audit ledger + PoW comparison + Fabric-style contract |
| Consensus analysis | Not explicit | PoA vs PoW measured |
| Privacy metrics | Accuracy / time | PER, ICR, PCR, LCR, VCR, InfCR, RPL |
| Best COVID accuracy | 97.25% | 98.00% centralized / 97.50% FedAvg / 96.75% ILA-CKKS best |

---

## 14. Main Findings

1. EfficientNet-B0 improves the original CNN baseline.
2. FedAvg works well for IID and moderate Non-IID data.
3. FedAvg fails under extreme Non-IID data.
4. FedDyn recovers much of the extreme Non-IID accuracy loss.
5. FedAvg is highly vulnerable to Byzantine sign-flip attacks.
6. Multi-Krum strongly improves Byzantine robustness.
7. CKKS can protect selected updates while preserving model utility.
8. Magnitude-only adaptive CKKS eventually saturates.
9. ILA-CKKS provides a stronger privacy-aware selection method.
10. Formal DP-SGD is possible but significantly reduces utility.
11. PoA blockchain is far more suitable than PoW for permissioned healthcare FL.
12. Fabric-style contracts improve the permissioned blockchain design.

---

## 15. Limitations

1. Brain MRI experiments are not currently documented.
2. The classifier-only update perturbation should not be presented as whole-model DP.
3. CKKS privacy metrics are proxy metrics, not formal leakage guarantees.
4. EXP-010 is not a live Hyperledger Fabric deployment.
5. The current gradient cosine field should be renamed or recomputed as true cosine similarity.
6. A final single-script end-to-end run combining Multi-Krum + ILA-CKKS + PoA blockchain would improve reproducibility.

---

## 16. Final Research Claim

The strongest accurate claim for this project is:

> This work modernizes and extends a 2025 blockchain-enabled FL-HE healthcare framework by introducing EfficientNet-B0, FedDyn, Multi-Krum Byzantine-resilient aggregation, leakage-aware selective CKKS encryption, formal and approximate privacy analysis, and a permissioned blockchain audit layer. The final framework preserves high COVID-19 classification performance while improving robustness, encryption practicality, privacy-awareness, and auditability.

---

## 17. Current Final Status

The project is now strong enough to be presented as a completed research prototype.

The only major missing item, if strict replication of the base paper is required, is Brain MRI validation.

Otherwise, the COVID-19 pipeline is complete with:

- model baseline
- federated learning
- Non-IID testing
- Byzantine defense
- homomorphic encryption
- privacy analysis
- blockchain audit
- consensus comparison
- Fabric-style permissioned contract abstraction
