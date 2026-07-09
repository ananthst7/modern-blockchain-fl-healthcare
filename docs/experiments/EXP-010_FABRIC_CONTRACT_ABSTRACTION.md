# EXP-010 — Hyperledger Fabric-style Smart Contract Abstraction

## Objective

Implement a Hyperledger Fabric-style smart contract abstraction for the federated healthcare blockchain system.

This experiment strengthens the blockchain part beyond a simple audit ledger by modeling the system as permissioned blockchain chaincode assets and functions.

## Motivation

The base paper uses hospitals, edge servers, encrypted local model updates, aggregation, and contribution rewards.

EXP-010 maps these into Fabric-style smart contract operations:

- RegisterHospital
- SubmitUpdateHash
- SubmitAggregation
- IssueRewards
- QueryRound
- QueryHospitalHistory
- VerifyContractState

## Why Fabric-style?

Hyperledger Fabric is suitable for consortium healthcare systems because participants are known organizations. This matches the cross-silo FL setting where hospitals and edge servers are permissioned entities.

This experiment does not deploy a full Fabric network. Instead, it implements the chaincode logic and world-state behavior in Python so it can be tested and documented inside the current PyTorch research repo.

## Assets

| Asset | Meaning |
|---|---|
| Hospital | Registered participating hospital |
| UpdateAsset | Hash record of a hospital's encrypted FL update |
| AggregationAsset | Record of selected hospitals and global model hash |
| RewardAsset | Contribution reward assigned to each hospital |

## Access Control

| Role | Allowed functions |
|---|---|
| Trusted authority | RegisterHospital |
| Hospital | SubmitUpdateHash |
| Edge server / validator | SubmitAggregation, IssueRewards |

## Stored On-Chain

Only audit metadata is stored:

- update hash
- encrypted update hash
- sample count
- local metric proxy
- aggregation hash
- global model hash
- contribution reward score

No raw patient data, raw images, raw model weights, or decrypted model updates are stored.

## Final EXP-010 Results

| Metric | Result |
|---|---:|
| Rounds | 5 |
| Hospitals | 4 |
| World state assets | 49 |
| History transactions | 49 |
| Avg register hospital time | 0.05685 ms |
| Avg submit update time | 0.04186 ms |
| Avg submit aggregation time | 0.05494 ms |
| Avg issue rewards time | 0.0954 ms |
| Avg query round time | 0.00426 ms |

The Fabric-style contract successfully modeled hospital registration, encrypted update-hash submission, aggregation records, reward issuance, round queries, and state verification.

## Comparison with Base Paper

| Component | Base Paper | EXP-010 |
|---|---|---|
| Blockchain tool | Ganache/Truffle | Fabric-style smart contract abstraction |
| Participants | Hospitals, edge servers, trusted authority | Same |
| Update record | Blockchain transaction | UpdateAsset |
| Aggregation record | Edge-server aggregation | AggregationAsset |
| Rewards | Contribution incentive | RewardAsset |
| Access control | Consortium assumption | Explicit role-based access control |
| Queryability | Not deeply evaluated | QueryRound and QueryHospitalHistory |
| Novelty | Blockchain + HE + FL | Fabric-compatible asset and chaincode design |

## Conclusion

EXP-010 adds a stronger permissioned-blockchain design layer to the project.

EXP-008 proves auditability.
EXP-009 proves PoA is better than PoW for this healthcare FL setting.
EXP-010 models the same system as Fabric-style smart contract assets with explicit access control and queryable world state.