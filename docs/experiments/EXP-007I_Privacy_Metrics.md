# EXP-007I --- Independent Privacy Metrics

## Purpose

CKKS has no epsilon-like privacy guarantee. We therefore introduce
independent validation metrics.

### PER

Encrypted parameters / Total parameters.

### AEQ

PCR / PER.

### PCR

Coverage computed using the selector's own ILA score.

### LCR

Fisher-only coverage: LCR = Selected Fisher / Total Fisher

Why trustworthy: Fisher Information is widely used as a parameter
importance proxy.

### VCR

Selected Gradient Variance / Total Gradient Variance.

### Influence Coverage Ratio

Selected (Update × Fisher) / Total (Update × Fisher).

### Gradient Energy Ratio

sqrt(Selected update energy / Total update energy).

Future work: replace with true cosine similarity.

## Interpretation

A good adaptive selector should achieve: - Low PER - High PCR - High
LCR - High VCR - High Influence Coverage - High accuracy - Low
communication overhead

## Threats to Validity

These are proxy metrics, not formal cryptographic guarantees. They
estimate protected information rather than proving indistinguishability.
