import copy
import torch


def initialize_h_states(global_state, num_clients):
    h_states = []

    for _ in range(num_clients):
        h = {}
        for key, value in global_state.items():
            if torch.is_floating_point(value):
                h[key] = torch.zeros_like(value.cpu())
        h_states.append(h)

    return h_states


def update_h_state(h_state, client_state, global_state, alpha):
    new_h = copy.deepcopy(h_state)

    for key in new_h.keys():
        if key in client_state and key in global_state:
            new_h[key] = (
                new_h[key]
                - alpha * (client_state[key].cpu() - global_state[key].cpu())
            )

    return new_h


def feddyn_aggregate(client_states, client_sizes, h_states, alpha):
    """
    FedDyn-style aggregation:
    weighted FedAvg plus dynamic correction from h states.
    """
    total_samples = sum(client_sizes)
    avg_state = copy.deepcopy(client_states[0])

    for key in avg_state.keys():
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = torch.zeros_like(avg_state[key])

            for state, size in zip(client_states, client_sizes):
                avg_state[key] += state[key] * (size / total_samples)

            h_avg = torch.zeros_like(avg_state[key])
            for h in h_states:
                if key in h:
                    h_avg += h[key] / len(h_states)

            avg_state[key] -= h_avg / alpha
        else:
            avg_state[key] = client_states[0][key]

    return avg_state