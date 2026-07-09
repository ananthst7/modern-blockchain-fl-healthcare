import math
import torch


def is_trainable_tensor_key(key: str) -> bool:
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False

    return key.endswith("weight") or key.endswith("bias")


def minmax_normalize(items, field):
    values = [item[field] for item in items]
    min_v = min(values)
    max_v = max(values)

    if abs(max_v - min_v) < 1e-12:
        for item in items:
            item[f"norm_{field}"] = 0.0
        return items

    for item in items:
        item[f"norm_{field}"] = (item[field] - min_v) / (max_v - min_v)

    return items


def update_historical_importance(historical_scores, current_scores, ema_alpha=0.2):
    for item in current_scores:
        key = item["key"]
        current = item["update_norm"]

        previous = historical_scores.get(key, current)
        historical_scores[key] = (1.0 - ema_alpha) * previous + ema_alpha * current

    return historical_scores


def compute_privacy_aware_scores(
    client_state,
    global_state,
    historical_scores,
    update_weight=0.5,
    history_weight=0.3,
    layer_weight=0.2,
):
    scores = []

    for key, tensor in client_state.items():
        if key not in global_state:
            continue

        if not torch.is_floating_point(tensor):
            continue

        if not is_trainable_tensor_key(key):
            continue

        client_tensor = tensor.detach().cpu().float()
        global_tensor = global_state[key].detach().cpu().float()

        if client_tensor.shape != global_tensor.shape:
            continue

        delta = client_tensor - global_tensor
        update_norm = torch.norm(delta).item()
        num_params = client_tensor.numel()
        layer_importance = math.log(num_params + 1)

        scores.append(
            {
                "key": key,
                "update_norm": update_norm,
                "historical_importance": historical_scores.get(key, update_norm),
                "layer_importance": layer_importance,
                "num_params": num_params,
            }
        )

    if not scores:
        return []

    scores = minmax_normalize(scores, "update_norm")
    scores = minmax_normalize(scores, "historical_importance")
    scores = minmax_normalize(scores, "layer_importance")

    for item in scores:
        item["score"] = (
            update_weight * item["norm_update_norm"]
            + history_weight * item["norm_historical_importance"]
            + layer_weight * item["norm_layer_importance"]
        )

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores


def get_privacy_aware_selected_keys(
    client_states,
    global_state,
    historical_scores,
    max_selected_bytes,
    top_k=None,
    ema_alpha=0.2,
    update_weight=0.5,
    history_weight=0.3,
    layer_weight=0.2,
):
    combined_scores = {}

    # First compute raw current scores for EMA update
    current_score_rows = []

    for state in client_states:
        for key, tensor in state.items():
            if key not in global_state:
                continue

            if not torch.is_floating_point(tensor):
                continue

            if not is_trainable_tensor_key(key):
                continue

            client_tensor = tensor.detach().cpu().float()
            global_tensor = global_state[key].detach().cpu().float()

            if client_tensor.shape != global_tensor.shape:
                continue

            update_norm = torch.norm(client_tensor - global_tensor).item()
            current_score_rows.append(
                {
                    "key": key,
                    "update_norm": update_norm,
                }
            )

    historical_scores = update_historical_importance(
        historical_scores=historical_scores,
        current_scores=current_score_rows,
        ema_alpha=ema_alpha,
    )

    for state in client_states:
        scores = compute_privacy_aware_scores(
            client_state=state,
            global_state=global_state,
            historical_scores=historical_scores,
            update_weight=update_weight,
            history_weight=history_weight,
            layer_weight=layer_weight,
        )

        for item in scores:
            key = item["key"]
            if key not in combined_scores:
                combined_scores[key] = {
                    "key": key,
                    "score": 0.0,
                    "num_params": item["num_params"],
                    "update_norm": 0.0,
                    "historical_importance": item["historical_importance"],
                    "layer_importance": item["layer_importance"],
                }

            combined_scores[key]["score"] += item["score"]
            combined_scores[key]["update_norm"] += item["update_norm"]

    ranked = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)

    selected_keys = []
    selected_bytes = 0

    for item in ranked:
        key = item["key"]

        if key not in global_state:
            continue

        tensor = global_state[key]
        tensor_bytes = tensor.numel() * tensor.element_size()

        if selected_bytes + tensor_bytes > max_selected_bytes:
            continue

        selected_keys.append(key)
        selected_bytes += tensor_bytes

        if top_k is not None and len(selected_keys) >= top_k:
            break

    if not selected_keys:
        raise ValueError("No privacy-aware CKKS tensors selected under the byte budget.")

    return selected_keys, selected_bytes, historical_scores, ranked