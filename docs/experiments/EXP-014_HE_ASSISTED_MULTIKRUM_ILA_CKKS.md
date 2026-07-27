# EXP-014 — HE-Assisted Multi-Krum with Functional ILA-CKKS Aggregation

## 1. Why EXP-014 Was Added

The previous fully integrated run demonstrated the complete stack:

```text
FedDyn + Multi-Krum + ILA-CKKS + PoA blockchain
```

However, in that earlier run CKKS was mainly a cryptographic benchmark path. It encrypted selected tensors, performed encrypted averaging, decrypted the result, and measured overhead, but the actual global model update was still produced from plaintext Multi-Krum aggregation.

EXP-014 corrects that weakness. It makes CKKS part of the functional model-update path while keeping Multi-Krum as the main robustness contribution.

## 2. Final EXP-014 Architecture

```text
EfficientNet-B0
+ FedDyn local optimization
+ client-side Information Leakage-Aware tensor scoring
+ CKKS encrypted selected tensor upload
+ HE-derived pairwise distance scores
+ Multi-Krum robust client selection
+ CKKS homomorphic weighted averaging of selected robust clients
+ trusted-authority aggregate decryption
+ functional global selected-tensor update
+ PoA blockchain audit ledger
```

Short name:

```text
HE-Assisted Multi-Krum + Functional ILA-CKKS
```

Suggested paper abbreviation:

```text
HE-MK-ILA-CKKS
```

## 3. Difference Between EXP-011 and EXP-014

| Item | EXP-011 Integrated Run | EXP-014 Corrected HE-Assisted Run |
|---|---|---|
| CKKS encryption performed | Yes | Yes |
| Encrypted weighted averaging performed | Yes | Yes |
| HE result used to update global model | No | Yes |
| Multi-Krum included | Yes | Yes |
| Multi-Krum distance source | Plaintext model states | HE-derived selected-tensor distances |
| Blockchain audit logging | Yes | Yes |
| Server-side plaintext selected-update access | Yes in practical aggregation path | No for selected encrypted-update path |
| Best description | Integrated benchmark | Functional HE-assisted privacy-robust pipeline |

## 4. Protocol Flow

### Step 1 — Local client training

Each hospital/client receives the current global model and performs local FedDyn training.

FedDyn objective:

```text
loss = cross_entropy_loss + (alpha / 2) * ||w - w_global||^2 - <h_i, w>
```

Where:

- `w` is the local model parameter vector.
- `w_global` is the previous global model.
- `h_i` is the FedDyn client dynamic state.
- `alpha` controls the strength of dynamic regularization.

### Step 2 — Byzantine attack simulation

One client is treated as malicious and applies a sign-flip attack:

```text
w_malicious = -attack_scale * w_client
```

Default setting:

```text
malicious_client_index = 0
attack_scale = 5.0
```

### Step 3 — Client-side ILA metadata computation

Each client computes local sensitivity metadata:

```text
ILA score = update_norm * fisher_score * gradient_variance
```

Where:

- `update_norm` measures how much a tensor changed from the global model.
- `fisher_score` is a Fisher-like proxy using mean squared gradients.
- `gradient_variance` measures instability of gradient norms across batches.

The server role does not need plaintext tensors to choose ILA keys. It receives score metadata.

### Step 4 — ILA budgeted tensor selection

The coordinator selects tensors under a byte budget:

```text
max_selected_bytes = 2,000,000
```

This usually encrypts roughly 12% of EfficientNet-B0 trainable parameters, depending on the selected tensors.

### Step 5 — CKKS encryption

Each client encrypts the selected tensors using CKKS.

Default CKKS parameters:

```text
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2^40
```

### Step 6 — HE-derived Multi-Krum distances

Instead of running Multi-Krum directly on plaintext updates, EXP-014 derives pairwise distances from encrypted selected tensors.

Preferred computation:

```text
EncDistance(i, j) = sum((Enc(w_i_selected) - Enc(w_j_selected))^2)
```

A trusted-authority role decrypts only the distance result required for Multi-Krum scoring.

### Step 7 — Multi-Krum robust selection

Multi-Krum uses the distance matrix to select robust clients.

For each client:

```text
score_i = sum of nearest (n - f - 2) distances
```

Default setting:

```text
n = 4 clients
f = 1 malicious client
num_selected = 2 clients
```

### Step 8 — Functional CKKS aggregation

Only the encrypted updates from Multi-Krum-selected clients are averaged:

```text
Enc(w_selected_avg) = weighted_average(Enc(w_selected_client_1), Enc(w_selected_client_2), ...)
```

The trusted authority decrypts the selected-tensor aggregate.

### Step 9 — Global model update

This is the critical EXP-014 fix:

```text
global_model[selected_keys] = Dec(HE_weighted_average(selected_client_ciphertexts))
```

Non-selected tensors remain from the previous global model.

### Step 10 — PoA blockchain audit

The blockchain records:

- client update hashes;
- encrypted update hashes;
- selected clients;
- selected key count;
- aggregation hash;
- global model hash;
- global metrics;
- rewards;
- validator identity;
- block hash;
- Merkle root.

The chain is then validated and tamper tested.

## 5. Valid Claims After EXP-014

After running EXP-014, the following claims are valid:

1. CKKS is used in the actual global model-update path for ILA-selected tensors.
2. Multi-Krum remains part of the functional pipeline.
3. Multi-Krum uses HE-derived selected-tensor distance scores.
4. Selected encrypted tensors are homomorphically averaged.
5. The decrypted HE aggregate is inserted into the global model.
6. The aggregation-server role does not directly inspect plaintext selected client tensors.
7. The PoA blockchain records tamper-evident audit logs for the integrated process.

## 6. Claims to Avoid

Do not claim:

1. Multi-Krum is performed completely inside CKKS with no revealed intermediate score.
2. The full EfficientNet-B0 model is encrypted tensor-by-tensor in every round.
3. The prototype removes the trusted-authority assumption.
4. No plaintext exists anywhere in the single-process experimental runtime.
5. The system provides formal MPC-level privacy against all colluding parties.

The correct claim is narrower and defensible:

```text
EXP-014 implements HE-assisted Multi-Krum, where robust client selection is based on
HE-derived selected-tensor distance scores and the selected ILA tensors are
functionally aggregated under CKKS before being inserted into the global model.
```

## 7. Trusted Authority Assumption

The trusted authority is a separate permissioned role from the aggregation server.

It performs limited decryption:

1. Pairwise distance scores used for Multi-Krum ranking.
2. Final selected-tensor aggregate used to update the global model.

It does not publish client updates.

In a real healthcare consortium, this could be:

- a hospital consortium authority;
- an ethics-approved research coordinator;
- a permissioned validator group;
- a threshold decryption committee.

A stronger future version could replace the single trusted authority with threshold decryption.

## 8. Privacy-Robustness Tradeoff

EXP-014 exists because there is a real technical conflict:

```text
CKKS supports addition and multiplication.
Multi-Krum requires comparison, ranking, and selection.
```

Pure CKKS does not naturally perform sorting or argmin selection efficiently.

Therefore, EXP-014 uses a hybrid design:

```text
Encrypted arithmetic for distance construction and aggregation
+
limited trusted decryption for distance scores
+
Multi-Krum selection over the resulting distance matrix
```

This is more realistic than claiming fully encrypted Multi-Krum without comparison support.

## 9. Expected Output Files

Running:

```powershell
python src\run_exp014_he_assisted_multikrum_ila_ckks.py
```

creates:

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

## 10. Recommended Commands

From repo root:

```powershell
python src\run_exp014_he_assisted_multikrum_ila_ckks.py
```

For a faster no-model-save run:

```powershell
python src\run_exp014_he_assisted_multikrum_ila_ckks.py --no-save-model
```

For CPU-only debugging:

```powershell
python src\run_exp014_he_assisted_multikrum_ila_ckks.py --cpu --global-rounds 1 --no-save-model
```

## 11. How to Interpret Results

If accuracy is lower than EXP-011, that is not automatic failure.

EXP-014 is stricter:

- EXP-011 updated the whole model using plaintext Multi-Krum.
- EXP-014 updates only ILA-selected tensors using the HE aggregate.

So EXP-014 trades some optimization freedom for stronger privacy-path correctness.

The most important success indicators are:

| Indicator | Meaning |
|---|---|
| `functional_he_update = true` | HE aggregate actually enters the model update |
| `server_plaintext_selected_update_access = false` | Server role does not inspect selected plaintext tensors |
| `selected_indices` excludes malicious client | HE-assisted Multi-Krum worked |
| `chain_valid = true` | PoA audit ledger remained valid |
| `tamper_detected = true` | Ledger detects manipulation |
| non-random accuracy/F1 | HE-selected update path is functional |

## 12. Final Paper Wording

Use this in the paper:

```text
To integrate Byzantine-resilient aggregation with homomorphic encryption, the
proposed framework introduces an HE-assisted Multi-Krum protocol. Clients encrypt
ILA-selected update tensors using CKKS. Pairwise distance scores for Multi-Krum
selection are derived from encrypted selected tensors, and a trusted authority
decrypts only the resulting distance scores required for robust ranking. The
selected encrypted client tensors are then homomorphically averaged, and the
decrypted selected-tensor aggregate is inserted into the global model update.
This design keeps Multi-Krum in the functional training path while avoiding
direct plaintext selected-update exposure to the aggregation server.
```

## 13. Limitations

1. The implementation is a single-process research prototype.
2. The trusted-authority role is simulated inside the same Python process.
3. Distance scores are revealed.
4. Selected tensor names and sizes are revealed.
5. Only ILA-selected tensors are functionally aggregated under CKKS.
6. Non-selected tensors are retained from the previous global model.
7. The system is not a formal secure multiparty computation implementation.
8. Full encrypted Multi-Krum with no trusted authority would require encrypted comparison, MPC, FHE bootstrapping, or threshold protocols.

## 14. Final Status Before Running

```text
Implementation file prepared.
Documentation file prepared.
Results pending local execution.
```

After the run, update this document with values from:

```text
results/he_assisted_multikrum_ila_ckks/covid_he_assisted_multikrum_ila_ckks.json
results/he_assisted_multikrum_ila_ckks/poa_audit_summary.json
```
