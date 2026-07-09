import math


def required_noise_std_for_epsilon(
    epsilon: float,
    clip_norm: float,
    delta: float = 1e-5,
    sensitivity_mode: str = "replace_one_update",
) -> float:
    sensitivity = 2 * clip_norm if sensitivity_mode == "replace_one_update" else clip_norm
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


if __name__ == "__main__":
    for clip in [100, 10, 1, 0.5, 0.1]:
        print(f"\nclip_norm={clip}")
        for eps in [10, 5, 2, 1]:
            sigma = required_noise_std_for_epsilon(eps, clip)
            print(f"  epsilon={eps}: noise_std={sigma}")