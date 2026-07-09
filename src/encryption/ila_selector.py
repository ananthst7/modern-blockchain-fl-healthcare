import torch


def is_trainable_tensor_key(key: str) -> bool:
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False
    return key.endswith("weight") or key.endswith("bias")


def safe_minmax(values):
    min_v = min(values)
    max_v = max(values)

    if abs(max_v - min_v) < 1e-12:
        return [0.0 for _ in values]

    return [(v - min_v) / (max_v - min_v) for v in values]


def compute_ila_scores(client_state, global_state, fisher_scores, gradvar_scores):
    rows = []

    for key, client_tensor in client_state.items():
        if key not in global_state:
            continue
        if not torch.is_floating_point(client_tensor):
            continue
        if not is_trainable_tensor_key(key):
            continue

        global_tensor = global_state[key]

        if client_tensor.shape != global_tensor.shape:
            continue

        client_tensor = client_tensor.detach().cpu().float()
        global_tensor = global_tensor.detach().cpu().float()

        update_norm = torch.norm(client_tensor - global_tensor).item()
        fisher = fisher_scores.get(key, 0.0)
        gradvar = gradvar_scores.get(key, 0.0)
        tensor_bytes = client_tensor.numel() * client_tensor.element_size()

        rows.append(
            {
                "key": key,
                "update_norm": update_norm,
                "fisher_score": fisher,
                "gradient_variance_score": gradvar,
                "tensor_bytes": tensor_bytes,
                "num_params": client_tensor.numel(),
            }
        )

    if not rows:
        return []

    norm_update = safe_minmax([r["update_norm"] for r in rows])
    norm_fisher = safe_minmax([r["fisher_score"] for r in rows])
    norm_gradvar = safe_minmax([r["gradient_variance_score"] for r in rows])

    for i, row in enumerate(rows):
        row["norm_update"] = norm_update[i]
        row["norm_fisher"] = norm_fisher[i]
        row["norm_gradvar"] = norm_gradvar[i]

        row["ila_score"] = (
            (row["norm_update"] + 1e-8)
            * (row["norm_fisher"] + 1e-8)
            * (row["norm_gradvar"] + 1e-8)
        )

        row["value_density"] = row["ila_score"] / max(row["tensor_bytes"], 1)

    return sorted(rows, key=lambda x: x["value_density"], reverse=True)


def select_ila_keys_under_budget(
    client_states,
    global_state,
    fisher_scores_by_client,
    gradvar_scores_by_client,
    max_selected_bytes,
):
    combined = {}

    for idx, state in enumerate(client_states):
        rows = compute_ila_scores(
            client_state=state,
            global_state=global_state,
            fisher_scores=fisher_scores_by_client[idx],
            gradvar_scores=gradvar_scores_by_client[idx],
        )

        for row in rows:
            key = row["key"]

            if key not in combined:
                combined[key] = {
                    "key": key,
                    "ila_score": 0.0,
                    "value_density": 0.0,
                    "tensor_bytes": row["tensor_bytes"],
                    "num_params": row["num_params"],
                    "update_norm": 0.0,
                    "fisher_score": 0.0,
                    "gradient_variance_score": 0.0,
                }

            combined[key]["ila_score"] += row["ila_score"]
            combined[key]["value_density"] += row["value_density"]
            combined[key]["update_norm"] += row["update_norm"]
            combined[key]["fisher_score"] += row["fisher_score"]
            combined[key]["gradient_variance_score"] += row["gradient_variance_score"]

    ranked = sorted(
        combined.values(),
        key=lambda x: x["value_density"],
        reverse=True,
    )

    selected_keys = []
    selected_bytes = 0

    for row in ranked:
        key = row["key"]
        tensor_bytes = row["tensor_bytes"]

        if selected_bytes + tensor_bytes > max_selected_bytes:
            continue

        selected_keys.append(key)
        selected_bytes += tensor_bytes

    if not selected_keys:
        raise ValueError("No ILA-CKKS tensors selected under byte budget.")

    return selected_keys, selected_bytes, ranked


def compute_ila_privacy_coverage(
    client_state,
    global_state,
    fisher_scores,
    gradvar_scores,
    selected_keys,
):
    rows = compute_ila_scores(
        client_state=client_state,
        global_state=global_state,
        fisher_scores=fisher_scores,
        gradvar_scores=gradvar_scores,
    )

    total_leakage = 0.0
    selected_leakage = 0.0

    for row in rows:
        leakage_score = row["ila_score"]
        total_leakage += leakage_score

        if row["key"] in selected_keys:
            selected_leakage += leakage_score

    eps = 1e-12

    privacy_coverage_ratio = selected_leakage / max(total_leakage, eps)
    residual_privacy_leakage = 1.0 - privacy_coverage_ratio

    return {
        "total_ila_leakage_score": total_leakage,
        "selected_ila_leakage_score": selected_leakage,
        "privacy_coverage_ratio": privacy_coverage_ratio,
        "residual_privacy_leakage": residual_privacy_leakage,
    }


def aggregate_ila_privacy_metrics(metric_rows):
    if not metric_rows:
        return {}

    output = {}

    for key in metric_rows[0].keys():
        values = [
            row[key]
            for row in metric_rows
            if isinstance(row.get(key), (int, float))
        ]

        if values:
            output[f"avg_{key}"] = sum(values) / len(values)
            output[f"min_{key}"] = min(values)
            output[f"max_{key}"] = max(values)

    return output
def compute_independent_coverage_metrics(
    client_state,
    global_state,
    fisher_scores,
    gradvar_scores,
    selected_keys,
):
    total_fisher = 0.0
    selected_fisher = 0.0

    total_gradvar = 0.0
    selected_gradvar = 0.0

    total_influence = 0.0
    selected_influence = 0.0

    total_delta_sq = 0.0
    selected_delta_sq = 0.0

    selected_set = set(selected_keys)

    for key, client_tensor in client_state.items():
        if key not in global_state:
            continue
        if not torch.is_floating_point(client_tensor):
            continue
        if not is_trainable_tensor_key(key):
            continue

        global_tensor = global_state[key]

        if client_tensor.shape != global_tensor.shape:
            continue

        client_tensor = client_tensor.detach().cpu().float()
        global_tensor = global_tensor.detach().cpu().float()

        delta = client_tensor - global_tensor
        update_norm = torch.norm(delta).item()
        delta_sq = torch.sum(delta.pow(2)).item()

        fisher = fisher_scores.get(key, 0.0)
        gradvar = gradvar_scores.get(key, 0.0)

        influence = update_norm * fisher

        total_fisher += fisher
        total_gradvar += gradvar
        total_influence += influence
        total_delta_sq += delta_sq

        if key in selected_set:
            selected_fisher += fisher
            selected_gradvar += gradvar
            selected_influence += influence
            selected_delta_sq += delta_sq

    eps = 1e-12

    leakage_coverage_ratio = selected_fisher / max(total_fisher, eps)
    variance_coverage_ratio = selected_gradvar / max(total_gradvar, eps)
    influence_coverage_ratio = selected_influence / max(total_influence, eps)

    gradient_cosine_similarity = (
        selected_delta_sq / max(total_delta_sq, eps)
    ) ** 0.5

    return {
        "leakage_coverage_ratio": leakage_coverage_ratio,
        "variance_coverage_ratio": variance_coverage_ratio,
        "influence_coverage_ratio": influence_coverage_ratio,
        "gradient_cosine_similarity": gradient_cosine_similarity,
    }


def aggregate_independent_coverage_metrics(metric_rows):
    if not metric_rows:
        return {}

    output = {}

    for key in metric_rows[0].keys():
        values = [
            row[key]
            for row in metric_rows
            if isinstance(row.get(key), (int, float))
        ]

        if values:
            output[f"avg_{key}"] = sum(values) / len(values)
            output[f"min_{key}"] = min(values)
            output[f"max_{key}"] = max(values)

    return output