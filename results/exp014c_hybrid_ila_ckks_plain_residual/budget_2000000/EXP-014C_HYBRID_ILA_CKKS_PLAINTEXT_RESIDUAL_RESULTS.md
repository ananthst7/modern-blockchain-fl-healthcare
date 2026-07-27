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
| Total runtime | 261.83 s |
| Device | cuda |

## Per-Round Results

| Round | Accuracy | F1 | Selected clients | Selected keys | HE upload MB | Crypto overhead |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 81.00% | 80.47% | [2, 3] | 102 | 156.8449 | 17.3347% |
| 2 | 89.50% | 89.41% | [2, 3] | 116 | 156.8444 | 17.2276% |
| 3 | 94.00% | 93.98% | [2, 3] | 108 | 156.8483 | 17.2835% |
| 4 | 94.50% | 94.48% | [2, 3] | 116 | 156.8498 | 17.3240% |
| 5 | 94.50% | 94.48% | [2, 3] | 105 | 156.8462 | 17.4976% |

## HE-Assisted Multi-Krum

The aggregation-server role uses distance scores derived from encrypted selected tensors. Multi-Krum ranks clients using these scores, then selected encrypted tensors are homomorphically averaged.

| Round | Selected indices | Distance mode counts | HE distance time |
|---:|---|---|---:|
| 1 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 5.4676 s |
| 2 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 5.4348 s |
| 3 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 5.5114 s |
| 4 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 5.5093 s |
| 5 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 6.9327 s |

## ILA-CKKS Coverage Metrics

| Round | PER | ICR | PCR | RPL | LCR | VCR | InfCR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.47% | 31.42% | 99.99% | 0.01% | 99.88% | 92.54% | 98.93% |
| 2 | 12.47% | 33.65% | 99.96% | 0.04% | 97.54% | 87.65% | 95.31% |
| 3 | 12.47% | 31.58% | 99.96% | 0.04% | 96.58% | 87.71% | 94.52% |
| 4 | 12.47% | 33.48% | 99.90% | 0.10% | 96.88% | 83.98% | 94.47% |
| 5 | 12.47% | 31.07% | 99.92% | 0.08% | 95.99% | 84.41% | 93.75% |

## Blockchain Audit Summary

| Metric | Value |
|---|---:|
| num_blocks_including_genesis | 6 |
| num_training_round_blocks | 5 |
| num_transactions | 31 |
| avg_block_creation_time_ms | 0.24813998024910688 |
| max_block_creation_time_ms | 0.2828999422490597 |
| verification_time_ms | 1.3766000047326088 |
| ledger_size_kb | 28.5654296875 |
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
