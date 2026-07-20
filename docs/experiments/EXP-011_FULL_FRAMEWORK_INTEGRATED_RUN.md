# EXP-011 — Full End-to-End Integrated Framework

## Objective

Run the final project architecture as one integrated pipeline instead of reporting separate components independently.

The final integrated architecture is:

```text
EfficientNet-B0
+ FedDyn local optimization
+ Multi-Krum Byzantine-resilient aggregation
+ ILA-CKKS selective homomorphic encryption
+ Proof-of-Authority blockchain audit logging
```

## Fixed Setting

| Parameter | Value |
|---|---|
| Dataset | COVID-19 Radiography Binary |
| Split | Moderate Non-IID |
| Clients | 4 |
| Global Rounds | 5 |
| Local Epochs | 1 |
| Learning Rate | 0.0001 |
| FedDyn Alpha | 0.005 |
| Model | EfficientNet-B0 |
| Aggregation | Multi-Krum |
| Attack | Sign-flip |
| Malicious Client Index | 0 |
| Attack Scale | 5.0 |
| CKKS Library | TenSEAL |
| CKKS Scheme | CKKS |
| ILA Plaintext Budget | 2,000,000 bytes |
| Blockchain | Proof-of-Authority audit ledger |
| Seed | 7 |
| Device | cuda |

## Round-Wise Results

| Round | Accuracy | F1 Score | Multi-Krum Selected Clients | ILA Keys | PER | ICR | PCR | LCR | VCR | InfCR | Enc Upload | Crypto Overhead | Round Time |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.00% | 81.78% | [2, 3] | 101 | 12.47% | 30.89% | 99.99% | 99.89% | 92.68% | 98.87% | 156.85 MB | 6.20% | 55.35 s |
| 2 | 89.75% | 89.68% | [2, 3] | 114 | 12.47% | 34.01% | 99.98% | 97.62% | 92.07% | 95.79% | 156.84 MB | 7.40% | 45.77 s |
| 3 | 95.25% | 95.24% | [2, 3] | 136 | 12.47% | 39.25% | 99.97% | 98.40% | 90.66% | 97.03% | 156.84 MB | 7.38% | 46.12 s |
| 4 | 95.75% | 95.74% | [2, 3] | 108 | 12.47% | 31.94% | 99.97% | 96.74% | 88.60% | 95.01% | 156.85 MB | 6.18% | 37.27 s |
| 5 | 96.75% | 96.75% | [2, 3] | 111 | 12.47% | 31.96% | 99.96% | 96.53% | 87.45% | 94.60% | 156.85 MB | 7.78% | 55.78 s |

## Final Result

| Metric | Value |
|---|---:|
| Final Accuracy | **96.75%** |
| Final F1 Score | **96.75%** |
| Best Accuracy | **96.75%** |
| Best F1 Score | **96.75%** |
| Best Round | 5 |
| Total Runtime | 240.30 s |
| Average Round Time | 48.06 s |
| Final Round Time | 55.78 s |

## Final ILA-CKKS Metrics

| Metric | Final Value |
|---|---:|
| Parameter Encryption Ratio (PER) | 12.47% |
| Information Coverage Ratio (ICR) | 31.96% |
| Privacy Coverage Ratio (PCR) | 99.96% |
| Residual Privacy Leakage (RPL) | 0.04% |
| Leakage Coverage Ratio (LCR) | 96.53% |
| Variance Coverage Ratio (VCR) | 87.45% |
| Influence Coverage Ratio (InfCR) | 94.60% |
| Selected Plaintext Budget | 1,999,992 bytes |
| Selected Key Count | 111 |
| Encrypted Upload | 156.85 MB |
| Crypto Overhead | 7.78% |

## Blockchain Audit Result

| Metric | Value |
|---|---:|
| Blockchain Type | Proof-of-Authority audit ledger |
| Consensus | Authorized rotating validators |
| Validators | 3 |
| Blocks including Genesis | 6 |
| Training Round Blocks | 5 |
| Transactions | 31 |
| Average Block Creation Time | 0.1874 ms |
| Max Block Creation Time | 0.2129 ms |
| Verification Time | 1.0344 ms |
| Ledger Size | 28.28 KB |
| Chain Valid | True |
| Tamper Detected | True |

## Interpretation

The integrated run confirms that the final framework works end-to-end.

The strongest final result is:

```text
Full Framework = FedDyn + Multi-Krum + ILA-CKKS + PoA Blockchain
Final Accuracy = 96.75%
Final F1 Score = 96.75%
```

Multi-Krum selected clients `[2, 3]` in every round, excluding the malicious sign-flip client from aggregation. ILA-CKKS encrypted approximately 12.47% of trainable parameters while covering 99.96% of the ILA-estimated privacy signal in the final round.

The blockchain layer did not change model accuracy. Its role is auditability, tamper evidence, contribution tracking, and traceability.
