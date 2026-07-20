# EXP-012 — Ablation Study

## Objective

Measure the contribution of each major component in the final framework under one fixed setting.

This experiment answers:

> What does each component add to the final healthcare federated learning system?

## Fixed Setting

| Parameter | Value |
|---|---|
| Dataset | COVID Radiography Binary |
| Split | Moderate Non-IID |
| Clients | 4 |
| Malicious Clients | 1 |
| Attack | sign-flip |
| Attack Scale | 5.0 |
| Global Rounds | 5 |
| Local Epochs | 1 |
| FedDyn Alpha | 0.005 |
| Seed | 7 |

## Ablation Results

| Method | Accuracy | F1 Score | Byzantine Defense | CKKS | ILA Selection | Blockchain | Main Purpose |
|---|---:|---:|---|---|---|---|---|
| FedAvg | 50.00% | 33.33% | No | No | No | No | Basic FL baseline under attack |
| FedDyn | 50.00% | 33.33% | No | No | No | No | Non-IID robustness under attack |
| FedDyn + Multi-Krum | 95.50% | 95.49% | Yes | No | No | No | Byzantine robustness |
| FedDyn + Multi-Krum + CKKS | 96.00% | 96.00% | Yes | Yes, classifier-only CKKS profiling | No | No | Encrypted update transmission profiling |
| FedDyn + Multi-Krum + ILA-CKKS | 96.75% | 96.75% | Yes | Yes | Yes | No | Privacy-aware selective encryption |
| Full Framework + Blockchain | 96.75% | 96.75% | Yes | Yes | Yes | Yes, PoA audit ledger | Auditable secure FL system |

## Key Interpretation

| Observation | Meaning |
|---|---|
| FedAvg under attack collapsed to 50.00% accuracy | Standard FedAvg cannot handle malicious sign-flip updates. |
| FedDyn alone also collapsed to 50.00% accuracy | Non-IID robustness alone is not enough when a Byzantine client is present. |
| FedDyn + Multi-Krum reached 95.50% accuracy | Robust aggregation is the key defense against malicious updates. |
| Adding CKKS reached 96.00% accuracy | Encryption/profiling remained compatible with robust aggregation. |
| Adding ILA-CKKS reached 96.75% accuracy | Leakage-aware encryption preserved model utility. |
| Full framework also reached 96.75% accuracy | Blockchain adds auditability, not model-learning improvement. |

## Paper-Safe Claim

The ablation confirms that the main accuracy recovery comes from Multi-Krum Byzantine defense, while ILA-CKKS adds privacy-aware selective encryption and the PoA blockchain adds auditability and tamper evidence.

The blockchain layer does not change classification accuracy because it does not participate in gradient computation or model aggregation.
