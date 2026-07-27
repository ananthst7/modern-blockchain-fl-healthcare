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
- Budget: 8,000,000 bytes
- Selected active keys: 109
- Selected active bytes: 7,999,992
- Active parameter encryption ratio: 0.498739
- Malicious client index: 0
- Attack: sign-flip, scale 5.0
- HE scheme: CKKS via TenSEAL
- Blockchain: PoA audit ledger enabled

## Final Result

| Metric | Value |
|---|---:|
| Final accuracy | 95.25% |
| Final precision | 95.44% |
| Final recall | 95.25% |
| Final F1 | 95.24% |
| Best accuracy | 95.25% |
| Best F1 | 95.24% |
| Final selected clients | [2, 3] |
| Final PER | 49.87% |
| Final PCR | 100.00% |
| Final encrypted upload | 623.56 MB |
| Final crypto overhead | 45.96% |
| Final round time | 81.51 s |

## Round-wise Result

| Round | Accuracy | F1 | Selected Clients | Active Keys | PER | Crypto Overhead | Round Time |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 68.00% | 64.60% | [2, 3] | 109 | 49.87% | 45.55% | 81.31s |
| 2 | 79.25% | 78.32% | [2, 3] | 109 | 49.87% | 45.28% | 81.26s |
| 3 | 89.50% | 89.38% | [2, 3] | 109 | 49.87% | 45.39% | 80.73s |
| 4 | 93.00% | 92.97% | [2, 3] | 109 | 49.87% | 45.53% | 80.43s |
| 5 | 95.25% | 95.24% | [2, 3] | 109 | 49.87% | 45.96% | 81.51s |

## Blockchain Audit Result

| Metric | Value |
|---|---:|
| Blocks including genesis | 6 |
| Training round blocks | 5 |
| Transactions | 31 |
| Average block creation time | 0.244860 ms |
| Verification time | 1.093100 ms |
| Ledger size | 28.4814 KB |
| Chain valid | True |
| Tamper detected | True |

## Correct Claim

EXP-014B supports the claim that ILA-selected tensors can be used as a functional active submodel: the selected tensors are the only locally trainable parameters, and 100% of that active submodel is CKKS-encrypted, used for HE-assisted Multi-Krum, homomorphically averaged, and inserted into the global model.

## Claim to Avoid

Do not claim that a small 2 MB / 4 MB / 8 MB budget updates the entire EfficientNet model. The correct framing is active-submodel learning, not full-model training under a small selective budget.

## Top ILA-ranked Tensors Used During Bootstrap

| Rank | Tensor | ILA Score | Value Density | Bytes | Params |
|---:|---|---:|---:|---:|---:|
| 1 | `classifier.1.bias` | 8.034123e-04 | 1.004265e-04 | 8 | 2 |
| 2 | `features.0.0.weight` | 1.948787e-02 | 5.638852e-06 | 3456 | 864 |
| 3 | `classifier.1.weight` | 1.899809e-02 | 1.855283e-06 | 10240 | 2560 |
| 4 | `features.1.0.block.0.0.weight` | 7.905020e-06 | 6.861997e-09 | 1152 | 288 |
| 5 | `features.1.0.block.2.0.weight` | 4.967692e-06 | 2.425631e-09 | 2048 | 512 |
| 6 | `features.1.0.block.0.1.weight` | 2.323014e-07 | 1.814854e-09 | 128 | 32 |
| 7 | `features.2.0.block.0.0.weight` | 2.000898e-06 | 3.256670e-10 | 6144 | 1536 |
| 8 | `features.1.0.block.1.fc1.weight` | 1.038990e-07 | 1.014639e-10 | 1024 | 256 |
| 9 | `features.2.0.block.3.0.weight` | 7.781563e-07 | 8.443536e-11 | 9216 | 2304 |
| 10 | `features.3.0.block.0.0.weight` | 1.133981e-06 | 8.202989e-11 | 13824 | 3456 |
| 11 | `features.2.0.block.1.0.weight` | 2.664459e-07 | 7.709661e-11 | 3456 | 864 |
| 12 | `features.1.0.block.1.fc2.bias` | 6.510819e-09 | 5.086578e-11 | 128 | 32 |
| 13 | `features.2.1.block.1.0.weight` | 2.576925e-07 | 4.970919e-11 | 5184 | 1296 |
| 14 | `features.2.1.block.0.0.weight` | 3.460018e-07 | 2.502907e-11 | 13824 | 3456 |
| 15 | `features.0.1.weight` | 2.841104e-09 | 2.219612e-11 | 128 | 32 |
| 16 | `features.1.0.block.2.1.weight` | 1.287641e-09 | 2.011939e-11 | 64 | 16 |
| 17 | `features.2.0.block.2.fc1.weight` | 2.845039e-08 | 1.852239e-11 | 1536 | 384 |
| 18 | `features.2.1.block.3.0.weight` | 2.477327e-07 | 1.792048e-11 | 13824 | 3456 |
| 19 | `features.3.0.block.3.0.weight` | 3.567892e-07 | 1.548564e-11 | 23040 | 5760 |
| 20 | `features.4.0.block.0.0.weight` | 5.553030e-07 | 1.446101e-11 | 38400 | 9600 |