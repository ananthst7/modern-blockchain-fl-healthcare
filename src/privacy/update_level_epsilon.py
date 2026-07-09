from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def gaussian_epsilon(noise_std: float, clip_norm: float, delta: float, sensitivity_mode: str) -> float:
    if noise_std <= 0:
        return float("inf")

    if sensitivity_mode == "replace_one_update":
        sensitivity = 2.0 * clip_norm
    elif sensitivity_mode == "add_remove_update":
        sensitivity = clip_norm
    else:
        raise ValueError("sensitivity_mode must be replace_one_update or add_remove_update")

    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / noise_std


def classify_privacy(epsilon: float) -> str:
    if epsilon <= 1:
        return "strong"
    if epsilon <= 10:
        return "moderate"
    if epsilon <= 100:
        return "weak"
    if epsilon <= 1000:
        return "very weak"
    return "negligible / utility-focused"


def amend_result_file(path: Path, noise_std: float, clip_norm: float, delta: float) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    eps_replace = gaussian_epsilon(
        noise_std=noise_std,
        clip_norm=clip_norm,
        delta=delta,
        sensitivity_mode="replace_one_update",
    )

    eps_add_remove = gaussian_epsilon(
        noise_std=noise_std,
        clip_norm=clip_norm,
        delta=delta,
        sensitivity_mode="add_remove_update",
    )

    data["update_level_dp_epsilon_accounting"] = {
        "dp_type": "classifier-only update-level Gaussian noise",
        "formal_status": "approximate Gaussian mechanism bound, not formal Opacus/RDP accounting",
        "delta": delta,
        "clip_norm": clip_norm,
        "noise_std": noise_std,
        "sensitivity_modes": {
            "replace_one_update": {
                "sensitivity": 2.0 * clip_norm,
                "epsilon": eps_replace,
                "privacy_grade": classify_privacy(eps_replace),
            },
            "add_remove_update": {
                "sensitivity": clip_norm,
                "epsilon": eps_add_remove,
                "privacy_grade": classify_privacy(eps_add_remove),
            },
        },
        "paper_safe_interpretation": (
            "This update-level DP configuration preserves model utility but does not provide "
            "a practically meaningful formal privacy guarantee because the Gaussian noise is extremely small "
            "relative to the clipping bound."
        ),
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] amended {path}")
    print(f"replace-one epsilon = {eps_replace:.6f}")
    print(f"add/remove epsilon = {eps_add_remove:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--noise-std", type=float, default=1e-6)
    parser.add_argument("--clip-norm", type=float, default=100.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()

    amend_result_file(
        path=Path(args.file),
        noise_std=args.noise_std,
        clip_norm=args.clip_norm,
        delta=args.delta,
    )


if __name__ == "__main__":
    main()