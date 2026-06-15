import copy
import torch


def flatten_state(state_dict):
    tensors = []

    for value in state_dict.values():
        if torch.is_floating_point(value):
            tensors.append(value.flatten().cpu())

    return torch.cat(tensors)


def squared_distance(state_a, state_b):
    vec_a = flatten_state(state_a)
    vec_b = flatten_state(state_b)
    return torch.sum((vec_a - vec_b) ** 2).item()


def multikrum(client_states, client_sizes, num_malicious=1, num_selected=2):
    """
    Multi-Krum robust aggregation.

    Selects the client updates closest to other updates and averages them.
    """
    n = len(client_states)
    f = num_malicious

    if n <= 2 * f + 2:
        print("Warning: Multi-Krum condition n > 2f + 2 not strictly satisfied.")

    scores = []

    for i in range(n):
        distances = []

        for j in range(n):
            if i != j:
                distances.append(squared_distance(client_states[i], client_states[j]))

        distances.sort()
        score = sum(distances[: max(1, n - f - 2)])
        scores.append((score, i))

    scores.sort(key=lambda x: x[0])
    selected_indices = [idx for _, idx in scores[:num_selected]]

    selected_states = [client_states[i] for i in selected_indices]
    selected_sizes = [client_sizes[i] for i in selected_indices]

    total_samples = sum(selected_sizes)
    avg_state = copy.deepcopy(selected_states[0])

    for key in avg_state.keys():
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = torch.zeros_like(avg_state[key])

            for state, size in zip(selected_states, selected_sizes):
                avg_state[key] += state[key] * (size / total_samples)
        else:
            avg_state[key] = selected_states[0][key]

    return avg_state, selected_indices, scores