# EXP-014 — HE-Assisted Multi-Krum with Functional ILA-CKKS Aggregation

## Purpose

EXP-014 fixes the previous sidecar CKKS limitation by inserting the decrypted homomorphic selected-tensor aggregate into the global model update path.

## Architecture

```text
EfficientNet-B0 + FedDyn + ILA selection + CKKS encrypted selected tensors
+ HE-derived Multi-Krum distances + Multi-Krum robust selection
+ CKKS weighted averaging + trusted aggregate decryption
+ functional global selected-tensor update + PoA blockchain
```

## Final Result

| Metric | Value |
|---|---:|
| Final accuracy | 94.50% |
| Final F1 | 94.48% |
| Best accuracy | 94.50% |
| Best F1 | 94.48% |
| Total runtime | 675.62 s |
| Device | cuda |

## Per-Round Results

| Round | Accuracy | F1 | Selected clients | Selected keys | HE upload MB | Crypto overhead |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 81.00% | 80.47% | [2, 3] | 213 | 1249.6736 | 56.0677% |
| 2 | 89.25% | 89.16% | [2, 3] | 213 | 1249.6430 | 56.3375% |
| 3 | 94.00% | 93.98% | [2, 3] | 213 | 1249.6791 | 56.2202% |
| 4 | 94.50% | 94.48% | [2, 3] | 213 | 1249.6856 | 56.6006% |
| 5 | 94.50% | 94.48% | [2, 3] | 213 | 1249.6787 | 56.4489% |

## HE-Assisted Multi-Krum

The aggregation-server role uses distance scores derived from encrypted selected tensors. Multi-Krum ranks clients using these scores, then selected encrypted tensors are homomorphically averaged.

| Round | Selected indices | Distance mode counts | HE distance time |
|---:|---|---|---:|
| 1 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 50.3739 s |
| 2 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 50.5714 s |
| 3 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 49.8934 s |
| 4 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 52.1122 s |
| 5 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 49.9279 s |

## ILA-CKKS Coverage Metrics

| Round | PER | ICR | PCR | RPL | LCR | VCR | InfCR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% |
| 2 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% |
| 3 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% |
| 4 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% |
| 5 | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% |

## Blockchain Audit Summary

| Metric | Value |
|---|---:|
| num_blocks_including_genesis | 6 |
| num_training_round_blocks | 5 |
| num_transactions | 31 |
| avg_block_creation_time_ms | 0.26395998429507017 |
| max_block_creation_time_ms | 0.2802999224513769 |
| verification_time_ms | 1.2024000752717257 |
| ledger_size_kb | 28.4375 |
| chain_valid | True |
| tamper_detected | True |

## Valid Claims

- CKKS is used in the functional selected-tensor global update path.
- Multi-Krum remains part of the functional training path.
- Robust selection uses HE-derived selected-tensor distance scores.
- Selected encrypted tensors are homomorphically averaged.
- The decrypted HE aggregate is inserted into the global model.
- BatchNorm/non-trainable state buffers are synchronized from the same Multi-Krum-selected clients to keep EfficientNet state consistent.
- The PoA ledger records hashes, selected clients, metrics and rewards.

## Claims to Avoid

- Fully encrypted Multi-Krum with no revealed score.
- Full-model encryption of every EfficientNet tensor.
- Removal of the trusted-authority assumption.
- Formal MPC-level privacy.
