# Implementation Decisions

## Baseline repository inspection

Official 2025 paper repository:
https://github.com/blockchainflhe/Blockchain-based-Federated-Learning-with-Homomorphic-Encryption

The repository was inspected and found to contain:
- FLHE.ipynb
- FLPyfhelin.py
- Flask backend
- React frontend
- Solidity contracts
- Ganache/Truffle setup

## Why we are not building directly on it

The reference implementation uses an outdated and mixed stack:
- TensorFlow/Keras for CNN training
- PyTorch 1.4.0 in backend requirements
- Syft
- old sklearn package name
- web3==5.9.0
- Ganache Ethereum setup
- notebook/prototype-style structure

This makes it difficult to extend cleanly with modern federated learning algorithms and PyTorch-based medical imaging models.

## What we will reuse conceptually

From FLPyfhelin.py:
- dataset preparation idea
- hospital/client splitting logic
- CNN baseline architecture
- encrypted weight aggregation concept
- Pyfhel/Microsoft SEAL usage pattern
- encryption time and aggregation time measurement

## Our clean implementation plan

We will build a new modular PyTorch-based implementation with:
- EfficientNet-B0
- FedAvg baseline
- FedDyn
- Multi-Krum Byzantine-resilient aggregation
- Differential Privacy
- CKKS encryption
- Proof-of-Authority blockchain simulation / Hyperledger-style logging
- COVID-19 Radiography Dataset
- Brain Tumor MRI Dataset

The official repository will remain only a reference baseline.