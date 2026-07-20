# EXP-013 — Scalability Study

## Objective

Evaluate how the final integrated framework scales as the number of logical hospitals increases.

The tested framework is:

```text
FedDyn + Multi-Krum + ILA-CKKS + PoA Blockchain
```

## Important Note About Logical Clients

If the requested number of clients exceeds the number of prepared client folders, existing client shards are reused cyclically as logical clients.

Therefore, this experiment measures system-level scalability: training overhead, encryption overhead, aggregation overhead, blockchain transaction growth, and ledger growth. It does not create new medical data diversity unless additional client splits are prepared.

## Scalability Results

| Clients | Accuracy | F1 Score | Avg Round Time | Avg Crypto Overhead | Final Enc Upload | Blockchain Tx Count | Ledger Size | Avg Block Creation | Tamper Detected |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 96.75% | 96.75% | 51.48 s | 6.99% | 156.85 MB | 31 | 28.29 KB | 0.2277 ms | True |
| 8 | 97.00% | 97.00% | 90.41 s | 7.61% | 313.69 MB | 51 | 45.67 KB | 0.3238 ms | True |
| 12 | 97.75% | 97.75% | 137.47 s | 7.45% | 470.54 MB | 71 | 63.68 KB | 0.7554 ms | True |

## Interpretation

The scalability study shows that the framework remains functional at 4, 8, and 12 logical clients.

Main trends:

1. Accuracy remained high across all tested client counts.
2. Average round time increased as more clients participated.
3. Encrypted upload scaled approximately linearly with client count.
4. Blockchain transaction count increased with the number of clients.
5. Ledger size increased with the number of logged client transactions.
6. PoA block creation time remained very small compared with training and encryption time.
7. Chain validation and tamper detection remained successful for all client counts.

## Paper-Safe Claim

The scalability study indicates that the main scaling cost comes from local training and encrypted upload size, while the PoA blockchain audit layer remains lightweight because it stores only hashes, metrics, selected-client metadata, and reward records.
