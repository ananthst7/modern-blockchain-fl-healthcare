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
| Total runtime | 480.20 s |
| Device | cuda |

## Per-Round Results

| Round | Accuracy | F1 | Selected clients | Selected keys | HE upload MB | Crypto overhead |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 81.00% | 80.47% | [2, 3] | 132 | 623.5552 | 42.3571% |
| 2 | 89.25% | 89.16% | [2, 3] | 152 | 623.5527 | 48.2416% |
| 3 | 94.00% | 93.98% | [2, 3] | 153 | 623.5521 | 37.2981% |
| 4 | 94.50% | 94.48% | [2, 3] | 165 | 623.5656 | 44.9295% |
| 5 | 94.50% | 94.48% | [2, 3] | 158 | 623.5459 | 38.8234% |

## HE-Assisted Multi-Krum

The aggregation-server role uses distance scores derived from encrypted selected tensors. Multi-Krum ranks clients using these scores, then selected encrypted tensors are homomorphically averaged.

| Round | Selected indices | Distance mode counts | HE distance time |
|---:|---|---|---:|
| 1 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 21.9330 s |
| 2 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 33.0839 s |
| 3 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 23.2635 s |
| 4 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 30.2510 s |
| 5 | [2, 3] | `{"he_square_sum_scalar_decryption": 6}` | 28.8720 s |

## ILA-CKKS Coverage Metrics

| Round | PER | ICR | PCR | RPL | LCR | VCR | InfCR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.87% | 56.35% | 100.00% | 0.00% | 99.94% | 97.21% | 99.53% |
| 2 | 49.87% | 60.84% | 99.99% | 0.01% | 98.44% | 96.13% | 97.44% |
| 3 | 49.87% | 61.32% | 99.99% | 0.01% | 98.27% | 96.46% | 97.57% |
| 4 | 49.87% | 64.58% | 99.99% | 0.01% | 98.78% | 95.68% | 98.11% |
| 5 | 49.87% | 64.56% | 99.99% | 0.01% | 98.64% | 95.95% | 98.11% |

## Blockchain Audit Summary

| Metric | Value |
|---|---:|
| num_blocks_including_genesis | 6 |
| num_training_round_blocks | 5 |
| num_transactions | 31 |
| avg_block_creation_time_ms | 0.27130001690238714 |
| max_block_creation_time_ms | 0.32570003531873226 |
| verification_time_ms | 1.3380999444052577 |
| ledger_size_kb | 28.5732421875 |
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
