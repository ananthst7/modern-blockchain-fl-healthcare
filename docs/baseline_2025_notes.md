# 2025 Baseline Paper Notes

Primary baseline:
Blockchain-based Federated Learning with Homomorphic Encryption for Privacy-Preserving Healthcare Data Sharing

Official GitHub:
https://github.com/blockchainflhe/Blockchain-based-Federated-Learning-with-Homomorphic-Encryption

Datasets:
1. COVID-19 Radiography Dataset
2. Brain Tumor MRI Dataset

Baseline components:
- CNN model
- Federated Learning
- Homomorphic Encryption using Pyfhel / Microsoft SEAL
- Blockchain using Ganache Ethereum
- Binary classification:
  - COVID vs Normal
  - Meningioma vs No Tumor

Our planned improvements:
- EfficientNet-B0 instead of CNN
- FedDyn instead of basic FedAvg
- Multi-Krum Byzantine-resilient aggregation
- Differential Privacy
- CKKS encryption
- Proof-of-Authority / Hyperledger-style blockchain layer
- Non-IID hospital simulation
- Communication, privacy, and computation cost analysis