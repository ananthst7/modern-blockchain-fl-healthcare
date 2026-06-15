import copy
import torch


def sign_flip_attack(client_state, scale=5.0):
    """
    Byzantine attack: flips and amplifies floating-point model updates.
    """
    attacked_state = copy.deepcopy(client_state)

    for key in attacked_state.keys():
        if torch.is_floating_point(attacked_state[key]):
            attacked_state[key] = -scale * attacked_state[key]

    return attacked_state