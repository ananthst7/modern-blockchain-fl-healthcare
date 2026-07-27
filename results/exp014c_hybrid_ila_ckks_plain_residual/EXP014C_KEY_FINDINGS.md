# EXP-014C Key Findings

- Hybrid aggregation preserved final accuracy at 94.50% for 2 MB, 4 MB and 8 MB budgets.
- The 2 MB hybrid budget is the best efficiency/utility point: it encrypts only 12.47% of trainable values while reaching 94.50% accuracy and 94.48% F1.
- The 8 MB hybrid budget is the strongest privacy-coverage point: it encrypts 49.87% of trainable values and lowers final residual privacy leakage to approximately 0.0096%.
- Multi-Krum consistently selected benign clients [2, 3] and excluded malicious client 0.
- The PoA audit chain remained valid and detected tampering in all budget runs.
- EXP-014C supports the final paper claim that selective ILA-CKKS can protect measured high-leakage tensors while plaintext residual aggregation preserves full-model utility.

## Recommended Result to Highlight

Highlight **EXP-014C at 2 MB** as the best practical result: 94.50% accuracy, 94.48% F1, 12.47% sensitive HE ratio, 99.9196% PCR, 17.50% final crypto overhead.

## Alternative Privacy-Heavy Result

Highlight **EXP-014C at 8 MB** when emphasizing privacy coverage: 49.87% sensitive HE ratio, 99.9904% PCR, 0.0096% residual privacy leakage, 94.50% accuracy.