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
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target, DefendedPAMPOSTarget
from src.attacks.adversarial_ml_attacks.knockoff_nets import KnockoffNetsAttack, pampos_target_query_func
from src.attacks.adversarial_ml_attacks.attribute_inference_black_box import (
    AttributeInferenceBlackBoxAttack, flatten_window_query_func,
)


def sample_smooth_windows(n, seq_len, feature_mean, feature_std, step_scale=0.15):
    base = np.random.normal(feature_mean, feature_std, size=(n, 1, len(feature_mean)))
    steps = np.random.normal(0.0, feature_std * step_scale, size=(n, seq_len, len(feature_mean)))
    trajectories = base + np.cumsum(steps, axis=1)
    return trajectories.reshape(n, -1).astype(np.float32)


def run_knockoff_against(target, seq_len, n_features, val_windows, label):
    input_dim = seq_len * n_features
    feature_mean = target.base_target.norm_mean.cpu().numpy() if hasattr(target, "base_target") else target.norm_mean.cpu().numpy()
    feature_std = target.base_target.norm_std.cpu().numpy() if hasattr(target, "base_target") else target.norm_std.cpu().numpy()

    query_func = pampos_target_query_func(target, seq_len, n_features)
    attack = KnockoffNetsAttack(input_dim=input_dim, output_dim=1, mode="classification", query_budget=500, strategy="adaptive")
    attack._sample_candidates = lambda n: sample_smooth_windows(n, seq_len, feature_mean, feature_std)

    x_eval = np.array([w.flatten() for w in val_windows[:100]])
    attack.extract(query_func, x_eval=x_eval, target_eval_func=query_func)

    print(f"[knockoff -- {label}] fidelity={attack.final_fidelity:.3f}, queries={attack.query_count}")
    return attack.final_fidelity


def run_attribute_inference_against(target, seq_len, n_features, train_windows, val_windows, label):
    input_dim = seq_len * n_features
    attribute_index = (seq_len - 1) * n_features + 6
    query_func = flatten_window_query_func(target, seq_len, n_features)

    aux_states = np.array([w.flatten() for w in train_windows[:300]])
    eval_states = np.array([w.flatten() for w in val_windows[:100]])

    attack = AttributeInferenceBlackBoxAttack(input_dim=input_dim, attribute_index=attribute_index, attribute_values=[0.0, 1.0])
    attack.fit(aux_states, query_func)
    results = attack.evaluate_accuracy(eval_states, query_func)

    print(f"[attribute inference -- {label}] accuracy={results['accuracy']:.3f} (baseline {results['baseline_accuracy']:.3f}), advantage={results['advantage_over_baseline']:+.3f}")
    return results


def check_utility_cost(base_target, defended_target, val_windows):
    print("\n" + "=" * 60)
    print("UTILITY COST CHECK -- does the defense hurt real detection?")
    print("=" * 60)

    agreements = 0
    base_flagged = 0
    defended_flagged = 0

    for w in val_windows:
        base_label = base_target.predict_label(w)
        defended_label = defended_target.predict_label(w)
        if base_label == defended_label:
            agreements += 1
        base_flagged += base_label
        defended_flagged += defended_label

    agreement_rate = agreements / len(val_windows)
    print(f"Decision agreement between defended and undefended detector on {len(val_windows)} real held-out windows: {agreement_rate:.3f}")
    print(f"Undefended flagged {base_flagged}/{len(val_windows)} | Defended flagged {defended_flagged}/{len(val_windows)}")
    print(f"NOTE: agreement < 1.0 means the defense's added noise sometimes changes real detection outcomes -- this is the honest cost of the defense, not zero-cost.")

    return {"agreement_rate": agreement_rate, "base_flagged": base_flagged, "defended_flagged": defended_flagged, "n_windows": len(val_windows)}


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

    print("[setup] Loading trained PAMPOS checkpoint...")
    base_target = load_pampos_target(REPO_ROOT, model_config)

    print("[setup] Rebuilding real train/val split...")
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    val_windows = [full_dataset.sequences[i] for i in val_subset.indices]

    print("[setup] Calibrating base (undefended) target...")
    base_target.calibrate(train_windows[:300])
    print(f"[setup] Threshold: {base_target.threshold:.4f}")

    defended_target = DefendedPAMPOSTarget(base_target, noise_std=base_target.threshold * 0.15, hard_label_only=True)
    print(f"[setup] Defended target: noise_std={defended_target.noise_std:.4f}, hard_label_only=True")

    results = {}

    print("\n" + "=" * 60)
    print("KNOCKOFF NETS -- before vs. after defense")
    print("=" * 60)
    fidelity_before = run_knockoff_against(base_target, seq_len, n_features, val_windows, "UNDEFENDED")
    fidelity_after = run_knockoff_against(defended_target, seq_len, n_features, val_windows, "DEFENDED")
    print(f"\n[summary] Knockoff fidelity: {fidelity_before:.3f} -> {fidelity_after:.3f} ({(fidelity_after - fidelity_before):+.3f})")
    results["knockoff_nets"] = {"fidelity_before": fidelity_before, "fidelity_after": fidelity_after}

    print("\n" + "=" * 60)
    print("ATTRIBUTE INFERENCE -- before vs. after defense")
    print("=" * 60)
    attr_before = run_attribute_inference_against(base_target, seq_len, n_features, train_windows, val_windows, "UNDEFENDED")
    attr_after = run_attribute_inference_against(defended_target, seq_len, n_features, train_windows, val_windows, "DEFENDED")
    print(f"\n[summary] Attribute inference advantage over baseline: {attr_before['advantage_over_baseline']:+.3f} -> {attr_after['advantage_over_baseline']:+.3f}")
    results["attribute_inference"] = {"before": attr_before, "after": attr_after}

    utility = check_utility_cost(base_target, defended_target, val_windows[:200])
    results["utility_cost"] = utility

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "defense_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[done] Saved results to {output_path}")


if __name__ == "__main__":
    main()