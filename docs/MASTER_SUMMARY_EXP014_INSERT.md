# Master Summary Insert — EXP-014

## EXP-014: HE-Assisted Multi-Krum with Functional ILA-CKKS Aggregation

EXP-014 was added to correct the main limitation of the previous integrated run. Earlier CKKS experiments measured encryption, encrypted averaging, decryption, communication expansion, and runtime overhead, but the actual global model update was produced by plaintext Multi-Krum. EXP-014 makes CKKS functional in the selected-tensor model-update path.

### Architecture

```text
EfficientNet-B0
+ FedDyn local optimization
+ ILA tensor selection
+ CKKS encrypted selected tensor upload
+ HE-derived Multi-Krum distance scores
+ Multi-Krum robust client selection
+ CKKS homomorphic aggregation of selected robust clients
+ trusted-authority aggregate decryption
+ functional global selected-tensor update
+ PoA blockchain audit ledger
```

### Scientific contribution

EXP-014 introduces a practical hybrid between Byzantine robustness and homomorphic encryption:

- CKKS protects ILA-selected update tensors during aggregation.
- Multi-Krum remains the Byzantine-resilient selection mechanism.
- Distance scores for Multi-Krum are derived from encrypted selected tensors.
- The HE aggregate is actually inserted into the global model.
- Blockchain records the process through tamper-evident audit metadata.

### Paper-safe final claim

```text
The proposed HE-assisted Multi-Krum protocol uses CKKS-encrypted ILA-selected
update tensors to derive robust-selection distance scores and to homomorphically
aggregate selected client updates. The decrypted selected-tensor aggregate is
used in the global model update, while the PoA ledger records hash pointers,
selected clients, metrics, and contribution rewards.
```

### Important limitation

EXP-014 is not a pure fully encrypted Multi-Krum protocol. Distance scores and selected-client indices are revealed, and the trusted authority decrypts limited outputs. This is a realistic permissioned-healthcare trust assumption, not a formal MPC protocol.
