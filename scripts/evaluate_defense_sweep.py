import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target_with_canonical_calibration, DefendedPAMPOSTarget
from src.attacks.adversarial_ml_attacks.attribute_inference_black_box import (
    AttributeInferenceBlackBoxAttack, flatten_window_query_func,
)

NOISE_MULTIPLIERS = [0.0, 0.15, 0.5, 1.0, 2.0, 4.0, 8.0]


def run_attribute_inference_against(target, seq_len, n_features, train_windows, val_windows):
    input_dim = seq_len * n_features
    attribute_index = (seq_len - 1) * n_features + 6
    query_func = flatten_window_query_func(target, seq_len, n_features)

    aux_states = np.array([w.flatten() for w in train_windows[:300]])
    eval_states = np.array([w.flatten() for w in val_windows[:100]])

    attack = AttributeInferenceBlackBoxAttack(input_dim=input_dim, attribute_index=attribute_index, attribute_values=[0.0, 1.0])
    attack.fit(aux_states, query_func)
    return attack.evaluate_accuracy(eval_states, query_func)


def check_utility_cost(base_target, defended_target, val_windows):
    agreements = 0
    for w in val_windows:
        if base_target.predict_label(w) == defended_target.predict_label(w):
            agreements += 1
    return agreements / len(val_windows)


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    n_features = config["model"]["input_dim"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint with canonical calibration...")
    base_target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    print(f"[setup] Canonical threshold: {base_target.threshold:.4f}")

    print("[setup] Rebuilding real train/val split (for train/val window access elsewhere in this script)...")
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    val_windows = [full_dataset.sequences[i] for i in val_subset.indices]

    utility_check_windows = val_windows[:200]

    print("\n" + "=" * 75)
    print(f"{'Noise (x thresh)':<18}{'Hard label':<12}{'Attr. accuracy':<16}{'Advantage':<12}{'Utility agreement':<18}")
    print("=" * 75)

    results = []
    for hard_label in [False, True]:
        for mult in NOISE_MULTIPLIERS:
            noise_std = base_target.threshold * mult
            defended = DefendedPAMPOSTarget(base_target, noise_std=noise_std, hard_label_only=hard_label)

            attr_results = run_attribute_inference_against(defended, seq_len, n_features, train_windows, val_windows)
            utility = check_utility_cost(base_target, defended, utility_check_windows) if noise_std > 0 else 1.0

            print(f"{mult:<18}{str(hard_label):<12}{attr_results['accuracy']:<16.3f}"
                  f"{attr_results['advantage_over_baseline']:<+12.3f}{utility:<18.3f}")

            results.append({
                "noise_multiplier": mult, "hard_label": hard_label,
                "accuracy": attr_results["accuracy"], "advantage": attr_results["advantage_over_baseline"],
                "utility_agreement": utility,
            })

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    good_defense = [r for r in results if r["advantage"] < 0.1 and r["utility_agreement"] > 0.9]
    if good_defense:
        best = min(good_defense, key=lambda r: r["advantage"])
        print(f"Best real tradeoff found: noise={best['noise_multiplier']}x threshold, "
              f"hard_label={best['hard_label']} -> advantage={best['advantage']:+.3f}, "
              f"utility agreement={best['utility_agreement']:.3f}")
    else:
        print("No setting in this sweep achieved both advantage < 0.1 AND utility agreement > 0.9.")
        print("This is a real finding: simple output perturbation may not be sufficient to defend")
        print("this specific leak without unacceptable cost to real detection accuracy.")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "defense_noise_sweep.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] Saved results to {output_path}")


if __name__ == "__main__":
    main()