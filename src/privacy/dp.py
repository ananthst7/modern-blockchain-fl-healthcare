import copy
import torch


def add_dp_noise_to_update(client_state, global_state, noise_std=1e-5, clip_norm=10.0):
    """
    DP-style clipping and Gaussian noise applied to model update delta:
    delta = client_state - global_state
    noisy_state = global_state + noisy_delta
    """
    noisy_state = copy.deepcopy(client_state)

    for key in client_state.keys():

    # Only apply DP to the classifier layer
        if "classifier" not in key:
            noisy_state[key] = client_state[key]
            continue

        if torch.is_floating_point(client_state[key]) and key in global_state:
            client_tensor = client_state[key].cpu()
            global_tensor = global_state[key].cpu()

            delta = client_tensor - global_tensor

            norm = torch.norm(delta)
            if norm > clip_norm:
                delta = delta * (clip_norm / (norm + 1e-8))

            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=delta.shape,
            )

            noisy_delta = delta + noise
            noisy_state[key] = global_tensor + noisy_delta
        else:
            noisy_state[key] = client_state[key]

    return noisy_state