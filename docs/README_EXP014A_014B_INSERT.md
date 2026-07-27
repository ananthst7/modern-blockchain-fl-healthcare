# README Insert — EXP-014A and EXP-014B

## EXP-014A: Full-trainable HE-assisted Multi-Krum

EXP-014A makes CKKS functional in the model-update path. The global model update is produced through HE-assisted Multi-Krum rather than plaintext Multi-Krum.

| Metric | Value |
|---|---:|
| Final accuracy | 94.50% |
| Final F1 | 94.48% |
| Best accuracy | 94.50% |
| Selected trainable keys | 213 |
| Parameter encryption ratio | 100.00% |
| Final encrypted upload | 1249.68 MB |
| Final crypto overhead | 56.45% |
| PoA chain valid | True |
| Tamper detected | True |

Correct framing: EXP-014A is the full-trainable HE correctness run. It proves the end-to-end encrypted aggregation path, but it is not the communication-efficient ILA setting because PER is 100%.

## EXP-014B: ILA-selected active-submodel HE-assisted Multi-Krum

EXP-014B evaluates true selective/adaptive encryption. The ILA-selected tensors define the active trainable submodel; non-selected trainable tensors remain frozen. This means every locally trained tensor is also CKKS-encrypted and HE-aggregated.

Run:

```powershell
python src\run_exp014b_active_submodel_ila_ckks.py --budgets 2000000 4000000 8000000
```

Quick test:

```powershell
python src\run_exp014b_active_submodel_ila_ckks.py --budgets 2000000 --global-rounds 1 --no-save-model
```
