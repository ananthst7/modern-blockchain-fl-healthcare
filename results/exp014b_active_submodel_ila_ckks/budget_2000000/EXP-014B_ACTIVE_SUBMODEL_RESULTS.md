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
- Budget: 2,000,000 bytes
- Selected active keys: 69
- Selected active bytes: 2,000,000
- Active parameter encryption ratio: 0.124685
- Malicious client index: 0
- Attack: sign-flip, scale 5.0
- HE scheme: CKKS via TenSEAL
- Blockchain: PoA audit ledger enabled

## Final Result

| Metric | Value |
|---|---:|
| Final accuracy | 76.50% |
| Final precision | 84.01% |
| Final recall | 76.50% |
| Final F1 | 75.13% |
| Best accuracy | 76.50% |
| Best F1 | 75.13% |
| Final selected clients | [2, 3] |
| Final PER | 12.47% |
| Final PCR | 100.00% |
| Final encrypted upload | 156.84 MB |
| Final crypto overhead | 20.34% |
| Final round time | 46.54 s |

## Round-wise Result

| Round | Accuracy | F1 | Selected Clients | Active Keys | PER | Crypto Overhead | Round Time |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 55.50% | 44.82% | [2, 3] | 69 | 12.47% | 20.30% | 47.50s |
| 2 | 56.00% | 45.44% | [2, 3] | 69 | 12.47% | 20.21% | 46.25s |
| 3 | 64.00% | 58.64% | [2, 3] | 69 | 12.47% | 20.83% | 47.90s |
| 4 | 69.75% | 66.70% | [2, 3] | 69 | 12.47% | 20.84% | 45.21s |
| 5 | 76.50% | 75.13% | [2, 3] | 69 | 12.47% | 20.34% | 46.54s |

## Blockchain Audit Result

| Metric | Value |
|---|---:|
| Blocks including genesis | 6 |
| Training round blocks | 5 |
| Transactions | 31 |
| Average block creation time | 0.256020 ms |
| Verification time | 1.101500 ms |
| Ledger size | 28.4639 KB |
| Chain valid | True |
| Tamper detected | True |

## Correct Claim

EXP-014B supports the claim that ILA-selected tensors can be used as a functional active submodel: the selected tensors are the only locally trainable parameters, and 100% of that active submodel is CKKS-encrypted, used for HE-assisted Multi-Krum, homomorphically averaged, and inserted into the global model.

## Claim to Avoid

Do not claim that a small 2 MB / 4 MB / 8 MB budget updates the entire EfficientNet model. The correct framing is active-submodel learning, not full-model training under a small selective budget.

## Top ILA-ranked Tensors Used During Bootstrap

| Rank | Tensor | ILA Score | Value Density | Bytes | Params |
|---:|---|---:|---:|---:|---:|
| 1 | `classifier.1.bias` | 8.003174e-04 | 1.000397e-04 | 8 | 2 |
| 2 | `features.0.0.weight` | 1.947297e-02 | 5.634540e-06 | 3456 | 864 |
| 3 | `classifier.1.weight` | 1.901043e-02 | 1.856487e-06 | 10240 | 2560 |
| 4 | `features.1.0.block.0.0.weight` | 7.898835e-06 | 6.856627e-09 | 1152 | 288 |
| 5 | `features.1.0.block.2.0.weight` | 4.964818e-06 | 2.424228e-09 | 2048 | 512 |
| 6 | `features.1.0.block.0.1.weight` | 2.319806e-07 | 1.812349e-09 | 128 | 32 |
| 7 | `features.2.0.block.0.0.weight` | 1.998816e-06 | 3.253282e-10 | 6144 | 1536 |
| 8 | `features.1.0.block.1.fc1.weight` | 1.038872e-07 | 1.014523e-10 | 1024 | 256 |
| 9 | `features.2.0.block.3.0.weight` | 7.779343e-07 | 8.441127e-11 | 9216 | 2304 |
| 10 | `features.3.0.block.0.0.weight` | 1.138564e-06 | 8.236141e-11 | 13824 | 3456 |
| 11 | `features.2.0.block.1.0.weight` | 2.660647e-07 | 7.698632e-11 | 3456 | 864 |
| 12 | `features.1.0.block.1.fc2.bias` | 6.485293e-09 | 5.066635e-11 | 128 | 32 |
| 13 | `features.2.1.block.1.0.weight` | 2.576940e-07 | 4.970949e-11 | 5184 | 1296 |
| 14 | `features.2.1.block.0.0.weight` | 3.458414e-07 | 2.501747e-11 | 13824 | 3456 |
| 15 | `features.0.1.weight` | 2.832279e-09 | 2.212718e-11 | 128 | 32 |
| 16 | `features.1.0.block.2.1.weight` | 1.276945e-09 | 1.995227e-11 | 64 | 16 |
| 17 | `features.2.0.block.2.fc1.weight` | 2.844953e-08 | 1.852183e-11 | 1536 | 384 |
| 18 | `features.2.1.block.3.0.weight` | 2.481740e-07 | 1.795240e-11 | 13824 | 3456 |
| 19 | `features.3.0.block.3.0.weight` | 3.571271e-07 | 1.550031e-11 | 23040 | 5760 |
| 20 | `features.4.0.block.0.0.weight` | 5.562898e-07 | 1.448671e-11 | 38400 | 9600 |