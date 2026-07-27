# EXP-014B — ILA-selected Active Submodel HE-assisted Multi-Krum

## Purpose

EXP-014B tests whether selective/adaptive encryption can remain functional in the actual global-model update path without pretending that a small encrypted subset can update a fully trainable model. The core design change is that the ILA-selected tensors define the active trainable submodel. Only those tensors are locally trained, encrypted, used for HE-assisted Multi-Krum distance scoring, homomorphically averaged, decrypted by the trusted authority, and inserted back into the global model.

## Configuration

- Dataset: COVID Radiography Binary
- Model: EfficientNet-B0
- Clients: 4
- Global rounds: 5
- Local epochs: 1
- FedDyn alpha: 0.005
- Budget: 4,000,000 bytes
- Selected active keys: 102
- Selected active bytes: 4,000,000
- Active parameter encryption ratio: 0.249370
- Malicious client index: 0
- Attack: sign-flip, scale 5.0
- HE scheme: CKKS via TenSEAL
- Blockchain: PoA audit ledger enabled

## Final Result

| Metric | Value |
|---|---:|
| Final accuracy | 88.75% |
| Final precision | 90.82% |
| Final recall | 88.75% |
| Final F1 | 88.61% |
| Best accuracy | 88.75% |
| Best F1 | 88.61% |
| Final selected clients | [2, 3] |
| Final PER | 24.94% |
| Final PCR | 100.00% |
| Final encrypted upload | 312.42 MB |
| Final crypto overhead | 31.25% |
| Final round time | 58.93 s |

## Round-wise Result

| Round | Accuracy | F1 | Selected Clients | Active Keys | PER | Crypto Overhead | Round Time |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 61.50% | 55.39% | [2, 3] | 102 | 24.94% | 30.94% | 60.08s |
| 2 | 64.75% | 59.75% | [2, 3] | 102 | 24.94% | 31.43% | 59.26s |
| 3 | 77.00% | 75.72% | [2, 3] | 102 | 24.94% | 31.52% | 58.76s |
| 4 | 84.25% | 83.85% | [2, 3] | 102 | 24.94% | 31.20% | 58.82s |
| 5 | 88.75% | 88.61% | [2, 3] | 102 | 24.94% | 31.25% | 58.93s |

## Blockchain Audit Result

| Metric | Value |
|---|---:|
| Blocks including genesis | 6 |
| Training round blocks | 5 |
| Transactions | 31 |
| Average block creation time | 0.253800 ms |
| Verification time | 1.226600 ms |
| Ledger size | 28.4922 KB |
| Chain valid | True |
| Tamper detected | True |

## Correct Claim

EXP-014B supports the claim that ILA-selected tensors can be used as a functional active submodel: the selected tensors are the only locally trainable parameters, and 100% of that active submodel is CKKS-encrypted, used for HE-assisted Multi-Krum, homomorphically averaged, and inserted into the global model.

## Claim to Avoid

Do not claim that a small 2 MB / 4 MB / 8 MB budget updates the entire EfficientNet model. The correct framing is active-submodel learning, not full-model training under a small selective budget.

## Top ILA-ranked Tensors Used During Bootstrap

| Rank | Tensor | ILA Score | Value Density | Bytes | Params |
|---:|---|---:|---:|---:|---:|
| 1 | `classifier.1.bias` | 8.082658e-04 | 1.010332e-04 | 8 | 2 |
| 2 | `features.0.0.weight` | 1.952670e-02 | 5.650088e-06 | 3456 | 864 |
| 3 | `classifier.1.weight` | 1.900423e-02 | 1.855882e-06 | 10240 | 2560 |
| 4 | `features.1.0.block.0.0.weight` | 7.897546e-06 | 6.855508e-09 | 1152 | 288 |
| 5 | `features.1.0.block.2.0.weight` | 4.983908e-06 | 2.433549e-09 | 2048 | 512 |
| 6 | `features.1.0.block.0.1.weight` | 2.340656e-07 | 1.828637e-09 | 128 | 32 |
| 7 | `features.2.0.block.0.0.weight` | 2.002813e-06 | 3.259787e-10 | 6144 | 1536 |
| 8 | `features.1.0.block.1.fc1.weight` | 1.040892e-07 | 1.016496e-10 | 1024 | 256 |
| 9 | `features.2.0.block.3.0.weight` | 7.792628e-07 | 8.455543e-11 | 9216 | 2304 |
| 10 | `features.3.0.block.0.0.weight` | 1.141560e-06 | 8.257812e-11 | 13824 | 3456 |
| 11 | `features.2.0.block.1.0.weight` | 2.667439e-07 | 7.718284e-11 | 3456 | 864 |
| 12 | `features.1.0.block.1.fc2.bias` | 6.566496e-09 | 5.130075e-11 | 128 | 32 |
| 13 | `features.2.1.block.1.0.weight` | 2.579842e-07 | 4.976546e-11 | 5184 | 1296 |
| 14 | `features.2.1.block.0.0.weight` | 3.461530e-07 | 2.504000e-11 | 13824 | 3456 |
| 15 | `features.0.1.weight` | 2.864558e-09 | 2.237936e-11 | 128 | 32 |
| 16 | `features.1.0.block.2.1.weight` | 1.299240e-09 | 2.030062e-11 | 64 | 16 |
| 17 | `features.2.0.block.2.fc1.weight` | 2.851656e-08 | 1.856547e-11 | 1536 | 384 |
| 18 | `features.2.1.block.3.0.weight` | 2.485759e-07 | 1.798148e-11 | 13824 | 3456 |
| 19 | `features.3.0.block.3.0.weight` | 3.572204e-07 | 1.550436e-11 | 23040 | 5760 |
| 20 | `features.4.0.block.0.0.weight` | 5.561980e-07 | 1.448432e-11 | 38400 | 9600 |