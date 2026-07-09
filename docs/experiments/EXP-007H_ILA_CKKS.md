# EXP-007H --- Information Leakage-Aware Adaptive CKKS (ILA-CKKS)

## Abstract

ILA-CKKS is a selective homomorphic encryption framework that
prioritizes encryption using update magnitude, Fisher Information and
gradient variance under a fixed communication budget.

## Motivation

Previous selective CKKS approaches ranked tensors only by update
magnitude. This work introduces an Information Leakage-Aware (ILA) score
to prioritize tensors that are estimated to leak the most information.

## ILA Score

ILA = \|\|ΔW\|\| × Fisher × GradientVariance

The selector: 1. Computes parameter updates. 2. Estimates Fisher
Information. 3. Estimates gradient variance. 4. Computes ILA scores. 5.
Sorts tensors. 6. Packs tensors under a byte budget. 7. Encrypts
selected tensors using CKKS.

## Threat Model

Honest-but-curious server capable of observing plaintext and encrypted
updates but without CKKS secret keys.

## Advantages

-   Privacy-aware selection.
-   Budget constrained.
-   Communication efficient.
-   Compatible with FedAvg, FedDyn and Multi-Krum.

## Limitations

-   Fisher and variance are proxy leakage measures.
-   Does not replace Differential Privacy.
-   Formal information-theoretic guarantees remain future work.

## Future Work

-   Recovery fidelity
-   Entropy-aware ranking
-   Influence functions
-   DP integration
