import copy
import torch


def fedavg(client_states, client_sizes):
    """
    Weighted average of client model parameters.
    Float tensors are averaged.
    Integer/Long tensors are copied from the first client.
    """
    total_samples = sum(client_sizes)
    avg_state = copy.deepcopy(client_states[0])

    for key in avg_state.keys():
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = torch.zeros_like(avg_state[key])

            for state, size in zip(client_states, client_sizes):
                avg_state[key] += state[key] * (size / total_samples)
        else:
            avg_state[key] = client_states[0][key]

    return avg_state