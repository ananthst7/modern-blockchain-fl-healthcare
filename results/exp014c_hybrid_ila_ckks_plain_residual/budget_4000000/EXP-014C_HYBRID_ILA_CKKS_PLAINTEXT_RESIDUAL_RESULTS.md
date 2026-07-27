# EXP-014C — Hybrid ILA-CKKS + Plaintext Residual Multi-Krum

## Purpose

EXP-014C implements a practical selective-encryption workflow: ILA-selected high-leakage tensors are aggregated through CKKS, while non-selected lower-risk residual tensors are aggregated in plaintext from the same Multi-Krum-selected clients.

## Architecture

```text
EfficientNet-B0 + FedDyn + ILA sensitive-tensor selection + CKKS selected tensors
+ HE-derived Multi-Krum distances + Multi-Krum robust selection
+ CKKS weighted averaging + trusted aggregate decryption
+ plaintext residual aggregation + full global model update + PoA blockchain
```

## Final Result

| Metric | Value |
|---|---:|
| Final accuracy | 94.50% |
| Final F1 | 94.48% |
| Best accuracy | 94.50% |
| Best F1 | 94.48% |
| Total runtime | 339.80 s |
| Device | cuda |

## Per-Round Results

| Round | Accuracy | F1 | Selected clients | Selected keys | HE upload MB | Crypto overhead |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 81.00% | 80.47% | [2, 3] | 143 | 312.4105 | 27.7300% |
| 2 | 89.25% | 89.16% | [2, 3] | 117 | 312.4045 | 27.3398% |
| 3 | 94.25% | 94.23% | [2, 3] | 140 | 312.4203 | 28.4639% |
| 4 | 94.50% | 94.48% | [2, 3] | 115 | 312.4090 | 28.2879% |
| 5 | 94.50% | 94.48% | [2, 3] | 139 | 312.4308 | 26.4308% |

## HE-Assisted Multi-Krum

The aggregation-server role uses distance scores derived from encrypted selected tensors. Multi-Krum ranks clients using these scores, then selected encrypted tensors are homomorphically averaged.

| Round | Selected indices | Distance mode counts | HE distance time |
|---:|---|---|---:|
| 1 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 13.5252 s |
| 2 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 10.7390 s |
| 3 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 10.8069 s |
| 4 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 13.4824 s |
| 5 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 12.3208 s |

## ILA-CKKS Coverage Metrics

| Round | PER | ICR | PCR | RPL | LCR | VCR | InfCR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.94% | 49.31% | 99.99% | 0.01% | 99.95% | 94.50% | 99.41% |
| 2 | 24.94% | 40.63% | 99.97% | 0.03% | 97.41% | 90.94% | 95.56% |
| 3 | 24.94% | 47.94% | 99.98% | 0.02% | 97.94% | 91.70% | 96.78% |
| 4 | 24.94% | 39.91% | 99.95% | 0.05% | 96.54% | 89.03% | 94.78% |
| 5 | 24.94% | 47.34% | 99.96% | 0.04% | 97.62% | 89.96% | 96.27% |

## Blockchain Audit Summary

| Metric | Value |
|---|---:|
| num_blocks_including_genesis | 6 |
| num_training_round_blocks | 5 |
| num_transactions | 31 |
| avg_block_creation_time_ms | 0.2578800078481436 |
| max_block_creation_time_ms | 0.2914000069722533 |
| verification_time_ms | 1.087700016796589 |
| ledger_size_kb | 28.5693359375 |
| chain_valid | True |
| tamper_detected | True |

## Valid Claims

- CKKS is used for the functional ILA-selected sensitive-tensor update path.
- Multi-Krum remains part of the functional training path.
- Robust selection uses HE-derived selected-tensor distance scores.
- Selected sensitive tensors are homomorphically averaged.
- The decrypted HE sensitive-tensor aggregate is merged with plaintext residual aggregation to update the full global model.
- Non-selected residual trainable tensors and BatchNorm/non-trainable buffers are aggregated/synchronized from the same Multi-Krum-selected clients.
- The PoA ledger records hashes, selected clients, metrics and rewards.

## Claims to Avoid

- Fully encrypted aggregation of every trainable tensor.
- Claiming that the non-selected residual tensors are hidden from the aggregation server.
- Removal of the trusted-authority assumption.
- Formal MPC-level privacy.
