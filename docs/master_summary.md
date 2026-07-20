# Master Summary — Modern Blockchain + Federated Learning + Homomorphic Encryption for Healthcare

## Project Status

This project implements a modernized and extended version of the 2025 base paper:

> **Blockchain-based Federated Learning with Homomorphic Encryption for Privacy-Preserving Healthcare Data Sharing**

The base paper used CNN + FedAvg + homomorphic encryption + Ganache/Truffle blockchain and reported **97.25% accuracy** on COVID-19 classification.

This project extends it with:

- EfficientNet-B0 instead of the original CNN.
- IID, moderate Non-IID, and extreme Non-IID federated settings.
- FedDyn for heterogeneous hospital data.
- Multi-Krum Byzantine-resilient aggregation.
- CKKS homomorphic encryption using TenSEAL.
- Selective, adaptive, budgeted, and Information Leakage-Aware CKKS.
- Approximate classifier-only update perturbation.
- Formal Opacus DP-SGD experiments.
- Proof-of-Authority blockchain audit ledger.
- PoA vs PoW blockchain consensus comparison.
- Hyperledger Fabric-style smart contract abstraction.

---

## Current Completion Status

| Component | Status | Notes |
|---|---|---|
| Centralized EfficientNet-B0 baseline | Completed | Best accuracy 98.00% |
| FedAvg IID | Completed | 97.50% accuracy |
| FedAvg moderate Non-IID | Completed | 97.00% accuracy |
| FedAvg extreme Non-IID | Completed | 82.00% accuracy |
| FedDyn extreme Non-IID | Completed | 92.25% best accuracy |
| Byzantine attack simulation | Completed | FedAvg collapses to 50.00% under sign-flip attack |
| Multi-Krum Byzantine defense | Completed | 97.00% IID, 93.50% moderate Non-IID |
| CKKS encryption | Completed | Fixed, adaptive, budgeted, and ILA-CKKS |
| ILA-CKKS | Completed | Best accuracy 96.75%, final 95.25% |
| CKKS independent privacy metrics | Completed | LCR 99.45%, VCR 86.27%, InfCR 97.34% |
| Approximate update perturbation | Completed | Classifier-only; not formal DP |
| Formal Opacus DP-SGD | Completed | Best utility 84.75% at ε = 173.04 |
| Blockchain audit ledger | Completed | PoA ledger, tamper detection true |
| Consensus comparison | Completed | PoW ≈ 1279.96× slower than PoA |
| Fabric-style smart contract abstraction | Completed | World state + access control + query functions |
| Brain MRI validation | Not completed / missing | Mention as future work unless results are added |
| Whole-model update-level DP | Not completed / invalidated | Current DP perturbation is classifier-only |

---

## Master Results Table

| Experiment | Setting | Method | Accuracy | F1 Score | Privacy / Security Notes |
|---|---|---|---:|---:|---|
| 2025 Base Paper | COVID-19 | CNN + FedAvg + HE + Blockchain | 97.25% | N/A | Pyfhel/SEAL + Ganache/Truffle |
| EXP-001 | Centralized | EfficientNet-B0 | **98.00%** | **98.00%** | Strong centralized baseline |
| EXP-002 | IID | FedAvg + EfficientNet-B0 | **97.50%** | **97.50%** | No encryption / no DP |
| EXP-003 | Moderate Non-IID | FedAvg | **97.00%** | **97.00%** | Moderate hospital heterogeneity |
| EXP-003B | Extreme Non-IID | FedAvg | **82.00%** | **82.00%** | Severe FedAvg degradation |
| EXP-004 | Extreme Non-IID | FedDyn, α = 0.01 | **88.00%** | **87.96%** | FedDyn improves over FedAvg |
| EXP-004A | Extreme Non-IID | FedDyn alpha sweep, α = 0.005 | **92.25%** | **92.25%** | Best extreme Non-IID result |
| EXP-005A | Extreme Non-IID + Byzantine | FedAvg + Sign-flip | 50.00% | 33.33% | Standard FedAvg collapses |
| EXP-005B | Extreme Non-IID + Byzantine | Multi-Krum | 60.25% | 52.79% | Partial recovery only |
| EXP-005C | Extreme Non-IID + Byzantine | FedDyn + Multi-Krum | 58.75% | 50.29% | Did not outperform Multi-Krum |
| EXP-005D | 8-client Extreme Non-IID + Byzantine | Multi-Krum | 50.50% | 34.43% | More clients alone did not solve heterogeneity |
| EXP-005E | IID + Byzantine | Multi-Krum | **97.00%** | **97.00%** | Strong Byzantine-resilient result |
| EXP-005F | Moderate Non-IID + Byzantine | Multi-Krum | **93.50%** | **93.47%** | Strong realistic Byzantine result |
| EXP-006A | IID | FedAvg + classifier-only update perturbation | **97.50%** | **97.50%** | Approx. ε ≈ 9.69×10⁸, not formal DP |
| EXP-006B | Extreme Non-IID | FedDyn + classifier-only update perturbation | **92.25%** | **92.25%** | Approx. ε ≈ 9.69×10⁸, not formal DP |
| EXP-006C | IID | FedAvg + Opacus DP-SGD | **84.75%** | **84.70%** | Formal DP, ε = 173.04 |
| EXP-006D | IID | Classifier-only privacy budget sweep | 87.50–89.75% | 87.49–89.75% | Approx. ε = 1–10, classifier-only study |
| EXP-007A | IID | FedAvg + fixed classifier-only CKKS | **97.75%** | **97.75%** | 1.275 MB encrypted upload, ~0.05% overhead |
| EXP-007B | Extreme Non-IID | FedDyn + fixed classifier-only CKKS | **90.75%** | **90.73%** | CKKS did not degrade FedDyn seed-7 path |
| EXP-007C | Moderate Non-IID + Byzantine | Multi-Krum + fixed classifier-only CKKS | **96.00%** | **96.00%** | Byzantine + CKKS compatible |
| EXP-007D | IID | FedAvg + adaptive CKKS | **97.50%** | **97.50%** | Top-k trainable tensor selection |
| EXP-007E | Moderate Non-IID + Byzantine | Multi-Krum + adaptive CKKS | **95.75%** | **95.74%** | Adaptive CKKS under Byzantine defense |
| EXP-007F | Moderate Non-IID + Byzantine | 1 MB budgeted adaptive CKKS | **94.75%** | **94.75%** | ICR 18.27%, PER 6.23%, upload 79.06 MB |
| EXP-007G | Moderate Non-IID + Byzantine | Adaptive CKKS budget sweep | **95.75% at 2 MB** | **95.74% at 2 MB** | Best magnitude-only ICR 32.43% |
| EXP-007H | Moderate Non-IID + Byzantine | ILA-CKKS algorithm | **96.75% best** | **96.75% best** | Leakage-aware tensor selection |
| EXP-007I | Moderate Non-IID + Byzantine | ILA-CKKS validation | **95.25% final** | **95.24% final** | LCR 99.45%, VCR 86.27%, InfCR 97.34% |
| EXP-008 | Blockchain | PoA audit ledger | N/A | N/A | Chain valid true, tamper detected true |
| EXP-009 | Blockchain | PoA vs PoW comparison | N/A | N/A | PoW ≈ 1279.96× slower than PoA |
| EXP-010 | Blockchain | Fabric-style contract abstraction | N/A | N/A | 49 world-state assets, 49 transactions |

---

## EXP-001 to EXP-004: Federated Learning Findings

### Centralized EfficientNet-B0

EfficientNet-B0 achieved **98.00% accuracy**, improving over the base paper's CNN result of **97.25%**.

### FedAvg

FedAvg performed well under IID and moderate Non-IID distributions:

| Setting | Accuracy |
|---|---:|
| IID | 97.50% |
| Moderate Non-IID | 97.00% |
| Extreme Non-IID | 82.00% |

Extreme Non-IID caused a major performance drop of **15.50 percentage points** compared with IID FedAvg.

### FedDyn

FedDyn addressed this heterogeneity problem.

| Method | Accuracy |
|---|---:|
| Extreme Non-IID FedAvg | 82.00% |
| FedDyn α = 0.01 | 88.00% |
| FedDyn α = 0.005 | **92.25%** |

FedDyn recovered **+10.25 percentage points** over extreme Non-IID FedAvg.

---

## EXP-005: Byzantine Robustness Findings

The project simulated a sign-flip Byzantine attack.

### Sign-flip Attack

A malicious client sends an inverted/scaled update:

```text
Malicious update = -attack_scale × original_update
```

where s = 5.0.

### Multi-Krum

Multi-Krum selects client updates whose distances to other updates are smallest, reducing the effect of outliers.

For each client update u_i, Multi-Krum computes distances to the closest benign-looking updates and selects the lowest-scoring updates.

### Results

| Setting | FedAvg + Attack | Multi-Krum + Attack |
|---|---:|---:|
| IID | 50.00% | **97.00%** |
| Moderate Non-IID | 50.00% | **93.50%** |
| Extreme Non-IID | 50.00% | 60.25% |

Conclusion: Multi-Krum is highly effective under IID and moderate Non-IID attack settings, but extreme heterogeneity remains difficult.

---

## EXP-006: Differential Privacy and Update Perturbation

### Important Clarification

The implemented update perturbation applies noise only to the **classifier layer**, not to the whole EfficientNet model. Therefore:

- It should not be claimed as whole-model update-level DP.
- It does not provide formal Opacus/RDP accounting.
- It is best described as **classifier-only Gaussian update perturbation**.

### Approximate Gaussian Mechanism Formula

```text
epsilon ≈ S × sqrt(2 × ln(1.25 / delta)) / sigma
```

where:

- S is sensitivity.
- delta = 10^-5.
- sigma is Gaussian noise standard deviation.
- For replace-one update sensitivity, S = 2C, where C is clip norm.
- For add/remove sensitivity, S = C.

For the original classifier-only perturbation:

| Parameter | Value |
|---|---:|
| Clip norm C | 100 |
| Noise std sigma | 1×10⁻⁶ |
| Delta | 1×10⁻⁵ |
| Replace-one ε | 968,961,052.52 |
| Add/remove ε | 484,480,526.26 |

This is **negligible / utility-focused privacy**.

### Classifier-only Privacy Budget Sweep

| Target ε | Accuracy | F1 Score | Interpretation |
|---:|---:|---:|---|
| 10 | 88.75% | 88.64% | Moderate approximate ε, classifier-only |
| 5 | 89.75% | 89.75% | Best sweep utility |
| 2 | 88.75% | 88.73% | Stronger approximate ε |
| 1 | 87.50% | 87.49% | Strong classifier-only perturbation |
| 0.1 | 87.50% | 87.48% | Suspicious / not paper-safe as formal DP |
| 0.00001 sanity | 85.00% | 84.76% | Confirms perturbation affects only classifier layer |

The ε = 0.1 and sanity-noise results should not be used to claim strong whole-model DP. They confirm that the EfficientNet feature extractor remains largely unaffected.

### Formal Opacus DP-SGD

| Configuration | Epsilon | Accuracy | Formal DP |
|---|---:|---:|---|
| Classifier-only DP, σ = 1.0 | 1.76 | 57.00% | Yes |
| Classifier-only DP, σ = 0.3, local epochs 3 | 50.50 | 75.25% | Yes |
| Classifier-only DP, σ = 0.2, local epochs 5 | 173.04 | **84.75%** | Yes |

Conclusion: Formal DP-SGD gives measurable privacy guarantees but with a large utility cost.

---

## EXP-007: CKKS Homomorphic Encryption

### CKKS Definition

CKKS is an approximate homomorphic encryption scheme that supports encrypted arithmetic over real-valued vectors. It allows model-update operations to be performed while selected update tensors remain encrypted.

### Homomorphic Property

```text
Decrypt(Encrypt(a) + Encrypt(b)) ≈ a + b
```

This enables secure aggregation of encrypted model components.

### CKKS Configuration

| Parameter | Value |
|---|---|
| Library | TenSEAL |
| Scheme | CKKS |
| Polynomial modulus degree | 8192 |
| Coeff modulus bits | [60, 40, 40, 60] |
| Global scale | 2^40 |
| Model | EfficientNet-B0 |
| Clients | 4 |
| Rounds | 5 |

### CKKS Experiment Progression

| Method | Accuracy | F1 | Encrypted Upload | Crypto Overhead | Key Finding |
|---|---:|---:|---:|---:|---|
| Fixed classifier CKKS | 97.75% | 97.75% | 1.275 MB | ~0.05% | Minimal overhead |
| Adaptive CKKS | 97.50% | 97.50% | 1.275 MB | ~0.05–0.07% | Dynamic selection works |
| 1 MB budget CKKS | 94.75% | 94.75% | 79.06 MB | 2.68–3.50% | Better privacy coverage |
| 2 MB magnitude-only CKKS | 95.75% | 95.74% | 156 MB | 4.78% | Best magnitude-only tradeoff |
| ILA-CKKS | 96.75% best / 95.25% final | 96.75% best / 95.24% final | 156.85 MB | 8.36% avg | Best privacy-aware selector |

---

## ILA-CKKS: Final Encryption Contribution

### Motivation

Earlier adaptive CKKS selected tensors by update magnitude:

```text
Score_i = ||Delta W_i||
```

This captures how much a tensor changed, but not whether it is privacy-sensitive.

### ILA Score

ILA-CKKS uses:

```text
ILA_i = ||Delta W_i|| × F_i × V_i
```

where:

- ||Delta W_i|| = update magnitude.
- F_i ≈ E[g_i^2] = Fisher-like sensitivity score.
- V_i = Var(||g_i||) = gradient variance score.

A tensor is prioritized only when it changes meaningfully, affects the loss, and has client-specific gradient variability.

### Budget-Constrained Selection

```text
maximize Sum of selected ILA scores
```

subject to:

```text
Sum of selected tensor bytes <= budget B
```

with B = 2,000,000 plaintext bytes.

### ILA-CKKS Results

| Metric | Mean | Final Round |
|---|---:|---:|
| Accuracy | 90.65% | 95.25% |
| Best Accuracy | 96.75% | Round 4 |
| F1 Score | 90.38% | 95.24% |
| Parameter Encryption Ratio | 12.47% | 12.47% |
| Information Coverage Ratio | 27.01% | 26.96% |
| Privacy Coverage Ratio | 99.95% | 99.93% |
| Leakage Coverage Ratio | 99.45% | 99.29% |
| Variance Coverage Ratio | 86.27% | 82.85% |
| Influence Coverage Ratio | 97.34% | 97.00% |
| Encrypted Upload | 156.848 MB | 156.852 MB |
| Crypto Overhead | 8.36% | 7.42% |

### ILA-CKKS Interpretation

ILA-CKKS encrypted only **12.47%** of trainable parameters but captured:

- **99.45%** Fisher-sensitive leakage signal.
- **86.27%** gradient-variance signal.
- **97.34%** update-weighted Fisher influence signal.

This is the strongest encryption contribution of the project.

---

## EXP-007I: Independent Privacy Metrics

CKKS does not have an epsilon-like DP parameter. Therefore, independent privacy coverage metrics were used.

| Metric | Formula | Meaning |
|---|---|---|
| PER | Encrypted Parameters / Total Parameters | How much of the model is encrypted |
| ICR | Sum of Selected Update Magnitudes / Sum of All Update Magnitudes | Update movement protected |
| PCR | Sum of Selected ILA Scores / Sum of All ILA Scores | ILA-estimated privacy signal protected |
| LCR | Sum of Selected Fisher Scores / Sum of All Fisher Scores | Fisher-sensitive signal protected |
| VCR | Sum of Selected Gradient Variances / Sum of All Gradient Variances | Gradient variance protected |
| InfCR | Sum of (Selected Update Magnitude × Fisher) / Sum of (All Update Magnitude × Fisher) | Update-sensitive Fisher influence protected |
| RPL | 1 - PCR | Residual privacy leakage |

The metrics are proxy metrics, not formal cryptographic leakage proofs.

---

## EXP-008 to EXP-010: Blockchain

### EXP-008 — PoA Blockchain Audit Ledger

A lightweight Proof-of-Authority blockchain audit ledger was implemented.

| Metric | Value |
|---|---:|
| Blocks including genesis | 6 |
| Training round blocks | 5 |
| Transactions | 31 |
| Clients / hospitals | 4 |
| Validators | 3 |
| Avg block creation time | 0.10874 ms |
| Max block creation time | 0.1379 ms |
| Verification time | 0.5587 ms |
| Ledger size | 28.37207 KB |
| Chain valid | true |
| Tamper detected | true |

The blockchain stores only hashes and metadata:

- local update hash
- encrypted update hash
- aggregation hash
- global model hash
- metrics
- contribution rewards

No raw patient data, images, gradients, or model weights are stored on-chain.

### EXP-009 — PoA vs PoW

| Metric | PoA | PoW |
|---|---:|---:|
| Blocks including genesis | 6 | 6 |
| Transactions | 31 | 31 |
| Avg block creation time | 0.09064 ms | 116.01598 ms |
| Max block creation time | 0.1048 ms | 260.7508 ms |
| Verification time | 0.4323 ms | 0.4595 ms |
| Chain valid | true | true |
| Tamper detected | true | true |

```text
Slowdown = 116.01598 / 0.09064 ≈ 1279.96
```

PoW was approximately **1279.96× slower** than PoA.

Conclusion: PoA is more suitable for healthcare FL because hospitals and edge validators are known permissioned participants.

### EXP-010 — Fabric-style Smart Contract Abstraction

The Fabric-style abstraction models:

- RegisterHospital
- SubmitUpdateHash
- SubmitAggregation
- IssueRewards
- QueryRound
- QueryHospitalHistory
- VerifyContractState

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

This is not a deployed Hyperledger Fabric network. It is a Fabric-style chaincode abstraction suitable for documenting the permissioned blockchain design.

---

## Best Results by Category

| Category | Best Result |
|---|---|
| Highest centralized accuracy | 98.00% — EfficientNet-B0 |
| Highest standard federated accuracy | 97.50% — FedAvg IID |
| Best extreme Non-IID result | 92.25% — FedDyn α = 0.005 |
| Best Byzantine-resilient result | 97.00% — Multi-Krum IID attack |
| Best realistic Byzantine result | 93.50% — Multi-Krum moderate Non-IID attack |
| Best CKKS utility result | 97.75% — fixed classifier CKKS |
| Best privacy-aware CKKS result | 96.75% best / 95.25% final — ILA-CKKS |
| Best formal DP utility | 84.75% — Opacus ε = 173.04 |
| Strongest formal DP | 57.00% — Opacus ε = 1.76 |
| Best blockchain design | PoA audit ledger + Fabric-style abstraction |
| Strongest final integrated direction | Multi-Krum + ILA-CKKS + PoA blockchain |

---

## Main Contributions Beyond the Base Paper

1. Replaced the original CNN with EfficientNet-B0.
2. Evaluated IID, moderate Non-IID, and extreme Non-IID hospital splits.
3. Demonstrated that FedAvg fails under extreme heterogeneity.
4. Added FedDyn for heterogeneous federated optimization.
5. Added Byzantine sign-flip attack simulations.
6. Added Multi-Krum Byzantine-resilient aggregation.
7. Replaced legacy HE tooling with TenSEAL CKKS.
8. Added fixed, adaptive, budgeted, and leakage-aware selective CKKS.
9. Proposed ILA-CKKS using update magnitude, Fisher sensitivity, and gradient variance.
10. Added independent CKKS privacy validation metrics.
11. Added approximate classifier-only privacy budget analysis.
12. Added formal Opacus DP-SGD experiments.
13. Implemented PoA blockchain audit ledger with tamper detection.
14. Compared PoA and PoW consensus.
15. Added Hyperledger Fabric-style smart contract abstraction.
16. Produced a modern PyTorch-based reproducible research codebase.

---

## Missing / Not Yet Complete Results

The following are still missing or should be treated as future work:

1. **Brain MRI validation**  
   The base paper includes COVID-19 and brain tumor MRI. This project currently documents COVID-19 binary classification results. Brain MRI experiments are not present in the uploaded logs.

2. **Whole-model update-level DP**  
   Current update perturbation is classifier-only. Whole-model update DP should be added only if stronger update-level privacy claims are required.

3. **True Hyperledger Fabric deployment**  
   EXP-010 is Fabric-style abstraction, not a deployed Docker Fabric network.

4. **True gradient cosine similarity**  
   The current cosine field in ILA logs behaves like a Gradient Energy Ratio. A true cosine metric should compute:
   ```text
cos(theta) = dot(g_selected, g_total) / (||g_selected|| × ||g_total||)
```

5. **Full end-to-end integrated run**  
   The final architecture is conceptually Multi-Krum + ILA-CKKS + PoA blockchain. If needed, a single script can run all three together and produce one final integrated result.

---

## Final Status

The project is substantially complete as a research prototype.

The strongest final claim is:

> This work modernizes the 2025 blockchain-FL-HE healthcare framework by replacing the CNN with EfficientNet-B0, adding robust FL optimization and Byzantine defense, introducing leakage-aware selective CKKS encryption, validating privacy coverage with independent metrics, and implementing a PoA/Fabric-style blockchain audit layer for transparent and tamper-evident federated learning.



---

## Final Integrated Experiments Added After EXP-010

The final project was completed with three additional experiments:

1. **EXP-011:** Full end-to-end integrated framework.
2. **EXP-012:** Ablation study.
3. **EXP-013:** Scalability study.

These experiments are important because they prove that the final architecture is not only a collection of separate modules, but a working integrated healthcare FL framework.

---

## EXP-011 — Full End-to-End Integrated Framework

Final architecture:

```text
EfficientNet-B0 + FedDyn + Multi-Krum + ILA-CKKS + PoA Blockchain
```

| Metric | Value |
|---|---:|
| Final Accuracy | **96.75%** |
| Final F1 Score | **96.75%** |
| Best Accuracy | **96.75%** |
| Best F1 Score | **96.75%** |
| Best Round | 5 |
| Global Rounds | 5 |
| Clients | 4 |
| Attack | Sign-flip, scale 5.0 |
| Malicious Client Index | 0 |
| Multi-Krum Selected Clients in Final Round | [2, 3] |
| Selected ILA Keys in Final Round | 111 |
| Final Encrypted Upload | 156.85 MB |
| Final Crypto Overhead | 7.78% |
| Total Runtime | 240.30 s |

### Final ILA-CKKS Privacy Coverage

| Metric | Final Value |
|---|---:|
| PER | 12.47% |
| ICR | 31.96% |
| PCR | 99.96% |
| RPL | 0.04% |
| LCR | 96.53% |
| VCR | 87.45% |
| InfCR | 94.60% |

### EXP-011 Blockchain Result

| Metric | Value |
|---|---:|
| Blocks including Genesis | 6 |
| Training Round Blocks | 5 |
| Transactions | 31 |
| Average Block Creation Time | 0.1874 ms |
| Verification Time | 1.0344 ms |
| Ledger Size | 28.28 KB |
| Chain Valid | True |
| Tamper Detected | True |

---

## EXP-012 — Final Ablation Study

Fixed setting: COVID-19 Radiography Binary, moderate Non-IID, 4 clients, 1 sign-flip malicious client, attack scale 5.0, 5 rounds, local epochs 1, FedDyn α = 0.005, seed 7.

| Method | Accuracy | F1 Score | Byzantine Defense | CKKS | ILA | Blockchain |
|---|---:|---:|---|---|---|---|
| FedAvg | 50.00% | 33.33% | No | No | No | No |
| FedDyn | 50.00% | 33.33% | No | No | No | No |
| FedDyn + Multi-Krum | 95.50% | 95.49% | Yes | No | No | No |
| FedDyn + Multi-Krum + CKKS | 96.00% | 96.00% | Yes | Yes, classifier-only CKKS profiling | No | No |
| FedDyn + Multi-Krum + ILA-CKKS | 96.75% | 96.75% | Yes | Yes | Yes | No |
| Full Framework + Blockchain | 96.75% | 96.75% | Yes | Yes | Yes | Yes, PoA audit ledger |

Key result: FedAvg and FedDyn alone collapsed to 50.00% accuracy under attack, while adding Multi-Krum restored accuracy to 95.50%. The full framework reached **96.75% accuracy** and **96.75% F1**.

---

## EXP-013 — Scalability Study

| Clients | Accuracy | F1 Score | Avg Round Time | Avg Crypto Overhead | Final Enc Upload | Blockchain Tx Count | Ledger Size | Tamper Detected |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 96.75% | 96.75% | 51.48 s | 6.99% | 156.85 MB | 31 | 28.29 KB | 0.2277 ms | True |
| 8 | 97.00% | 97.00% | 90.41 s | 7.61% | 313.69 MB | 51 | 45.67 KB | 0.3238 ms | True |
| 12 | 97.75% | 97.75% | 137.47 s | 7.45% | 470.54 MB | 71 | 63.68 KB | 0.7554 ms | True |

Important interpretation: client counts above 4 reuse the available prepared client folders cyclically. Therefore, EXP-013 measures system-level scalability rather than new medical-data diversity.

---

## Updated Final Status After EXP-011 to EXP-013

| Component | Status | Final Note |
|---|---|---|
| Full integrated framework | Completed | FedDyn + Multi-Krum + ILA-CKKS + PoA blockchain |
| Ablation study | Completed | Shows contribution of each component |
| Scalability study | Completed | Tested 4, 8, and 12 logical clients |
| Final full-framework accuracy | Completed | **96.75%** |
| Final full-framework F1 | Completed | **96.75%** |
| Blockchain tamper detection | Completed | True |
| Brain MRI validation | Still missing | Future work unless additional results are added |
| Whole-model update-level DP | Still missing | Current perturbation is classifier-only |
