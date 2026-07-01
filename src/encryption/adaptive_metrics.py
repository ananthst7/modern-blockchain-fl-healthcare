import torch


def is_trainable_tensor_key(key: str) -> bool:
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False

    return key.endswith("weight") or key.endswith("bias")


def compute_adaptive_encryption_metrics(
    client_state,
    global_state,
    selected_keys,
    fixed_keys=None,
):
    """
    Computes adaptive selective encryption quality metrics.

    UCR  = update coverage ratio
    PER  = parameter encryption ratio
    AEQ  = adaptive encryption quality = UCR / PER
    ILR  = information leakage ratio = 1 - UCR
    RRS  = reconstruction risk score = visible update norm / total update norm
    AG   = adaptive gain over fixed selector
    """

    fixed_keys = fixed_keys or []

    total_update_norm = 0.0
    selected_update_norm = 0.0
    fixed_update_norm = 0.0

    total_params = 0
    selected_params = 0
    fixed_params = 0

    tensor_scores = []

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
        num_params = client_tensor.numel()

        total_update_norm += update_norm
        total_params += num_params

        tensor_scores.append(
            {
                "key": key,
                "update_norm": update_norm,
                "num_params": num_params,
            }
        )

        if key in selected_keys:
            selected_update_norm += update_norm
            selected_params += num_params

        if key in fixed_keys:
            fixed_update_norm += update_norm
            fixed_params += num_params

    eps = 1e-12

    update_coverage_ratio = selected_update_norm / max(total_update_norm, eps)
    parameter_encryption_ratio = selected_params / max(total_params, eps)
    adaptive_encryption_quality = update_coverage_ratio / max(parameter_encryption_ratio, eps)

    information_leakage_ratio = 1.0 - update_coverage_ratio
    reconstruction_risk_score = information_leakage_ratio

    fixed_update_coverage_ratio = fixed_update_norm / max(total_update_norm, eps)
    fixed_parameter_encryption_ratio = fixed_params / max(total_params, eps)

    adaptive_gain = update_coverage_ratio - fixed_update_coverage_ratio

    tensor_scores = sorted(
        tensor_scores,
        key=lambda x: x["update_norm"],
        reverse=True,
    )

    rank_map = {
        item["key"]: rank + 1
        for rank, item in enumerate(tensor_scores)
    }

    selected_ranks = [
        rank_map[key]
        for key in selected_keys
        if key in rank_map
    ]

    avg_selected_rank = (
        sum(selected_ranks) / len(selected_ranks)
        if selected_ranks
        else None
    )

    return {
        "total_update_norm": total_update_norm,
        "selected_update_norm": selected_update_norm,
        "fixed_update_norm": fixed_update_norm,

        "total_trainable_params": total_params,
        "selected_encrypted_params": selected_params,
        "fixed_encrypted_params": fixed_params,

        "update_coverage_ratio": update_coverage_ratio,
        "parameter_encryption_ratio": parameter_encryption_ratio,
        "adaptive_encryption_quality": adaptive_encryption_quality,

        "information_leakage_ratio": information_leakage_ratio,
        "reconstruction_risk_score": reconstruction_risk_score,

        "fixed_update_coverage_ratio": fixed_update_coverage_ratio,
        "fixed_parameter_encryption_ratio": fixed_parameter_encryption_ratio,
        "adaptive_gain_over_fixed": adaptive_gain,

        "avg_selected_rank": avg_selected_rank,
        "selected_ranks": selected_ranks,
    }


def aggregate_adaptive_metrics(client_metrics):
    if not client_metrics:
        return {}

    keys = client_metrics[0].keys()
    output = {}

    for key in keys:
        values = [
            item[key]
            for item in client_metrics
            if isinstance(item.get(key), (int, float))
        ]

        if values:
            output[f"avg_{key}"] = sum(values) / len(values)
            output[f"min_{key}"] = min(values)
            output[f"max_{key}"] = max(values)

    return output