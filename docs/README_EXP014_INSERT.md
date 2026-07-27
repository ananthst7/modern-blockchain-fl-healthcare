## EXP-014 — HE-Assisted Multi-Krum with Functional ILA-CKKS Aggregation

EXP-014 is the corrected end-to-end privacy-robustness experiment. It addresses the limitation of earlier CKKS experiments where encrypted aggregation was measured as a sidecar path while the actual global model was updated using plaintext Multi-Krum.

EXP-014 uses the following pipeline:

```text
EfficientNet-B0
+ FedDyn local optimization
+ client-side ILA metadata generation
+ CKKS encryption of selected ILA tensors
+ HE-derived pairwise distance scores
+ Multi-Krum robust client selection
+ CKKS weighted averaging of selected robust clients
+ trusted-authority aggregate decryption
+ functional selected-tensor global model update
+ PoA blockchain audit ledger
```

The key correction is:

```text
global_model[selected_keys] = decrypted_HE_aggregate[selected_keys]
```

This means CKKS is used in the actual model-update path for ILA-selected tensors.

### EXP-014 claims

Valid:

- Multi-Krum remains part of the functional training path.
- Selected tensors are encrypted using CKKS.
- Pairwise Multi-Krum distance scores are derived from encrypted selected tensors.
- Selected encrypted tensors are homomorphically averaged.
- The decrypted HE aggregate is used to update the global model.
- The PoA ledger audits update hashes, encrypted hashes, selected clients, metrics, and rewards.

Not claimed:

- Full-model encrypted aggregation for every EfficientNet tensor.
- Fully encrypted Multi-Krum with no revealed score.
- Removal of the trusted-authority assumption.
- Formal MPC-level privacy.

Run:

```powershell
python src\run_exp014_he_assisted_multikrum_ila_ckks.py
```

Outputs:

```text
results/he_assisted_multikrum_ila_ckks/
    covid_he_assisted_multikrum_ila_ckks.json
    covid_he_assisted_multikrum_ila_ckks.pth
    round_history.csv
    poa_audit_ledger.json
    poa_audit_summary.json
    tamper_test.json
    EXP-014_HE_ASSISTED_MULTIKRUM_ILA_CKKS_RESULTS.md
```
