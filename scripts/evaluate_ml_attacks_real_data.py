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
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target_with_canonical_calibration
from src.attacks.adversarial_ml_attacks.membership_inference_black_box import run_membership_inference_v2
from src.attacks.adversarial_ml_attacks.knockoff_nets import KnockoffNetsAttack, pampos_target_query_func
from src.attacks.adversarial_ml_attacks.attribute_inference_black_box import (
    AttributeInferenceBlackBoxAttack, flatten_window_query_func,
)
from src.attacks.adversarial_ml_attacks.miface import MIFaceAttack
from src.attacks.adversarial_ml_attacks.hopskipjump import HopSkipJumpAttack, make_pampos_decision_func
from src.attacks.adversarial_ml_attacks.database_reconstruction import DatabaseReconstructionAttack, DatabaseTargetModel

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]


def get_real_train_val_windows(config, repo_root: Path):
    seq_len = config["training"]["seq_len"]
    data_root = repo_root / config["data"]["raw_dir"]

    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)

    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    val_windows = [full_dataset.sequences[i] for i in val_subset.indices]
    return train_windows, val_windows


def compute_empirical_bounds(windows: list, seq_len: int, n_features: int, lower_percentile: float = 2.0, upper_percentile: float = 98.0, padding_factor: float = 1.1) -> np.ndarray:
    stacked = np.concatenate([w.reshape(-1, n_features) for w in windows], axis=0)
    lows = np.percentile(stacked, lower_percentile, axis=0)
    highs = np.percentile(stacked, upper_percentile, axis=0)
    ranges = highs - lows
    padded_lows = lows - ranges * (padding_factor - 1.0) / 2.0
    padded_highs = highs + ranges * (padding_factor - 1.0) / 2.0
    per_feature_bounds = np.stack([padded_lows, padded_highs], axis=1)
    return np.tile(per_feature_bounds, (seq_len, 1))


def run_membership_inference(target, train_windows: list, val_windows: list, seq_len: int, n_features: int, results: dict):
    print("\n" + "=" * 60)
    print("MEMBERSHIP INFERENCE -- direct real-score comparison (corrected methodology)")
    print("=" * 60)

    from src.attacks.adversarial_ml_attacks.membership_inference_black_box import run_membership_inference_v2

    n_eval = min(200, len(train_windows), len(val_windows))
    member_sample = train_windows[:n_eval]
    nonmember_sample = val_windows[:n_eval]

    eval_results = run_membership_inference_v2(target, member_sample, nonmember_sample)
    print(f"Real accuracy: {eval_results['accuracy']:.3f} (baseline 0.5)")
    print(f"Balanced accuracy: {eval_results['balanced_accuracy']:.3f}")
    print(f"Advantage over baseline: {eval_results['advantage_over_baseline']:+.3f}")
    print(f"Member recall: {eval_results['member_recall']:.3f} | Nonmember recall: {eval_results['nonmember_recall']:.3f}")
    print(f"Calibration-stage accuracy (best achievable on calibration half): {eval_results['calibration_accuracy']:.3f}")

    results["membership_inference"] = eval_results


def sample_smooth_windows(n: int, seq_len: int, feature_mean: np.ndarray, feature_std: np.ndarray, step_scale: float = 0.15) -> np.ndarray:
    base = np.random.normal(feature_mean, feature_std, size=(n, 1, len(feature_mean)))
    steps = np.random.normal(0.0, feature_std * step_scale, size=(n, seq_len, len(feature_mean)))
    trajectories = base + np.cumsum(steps, axis=1)
    return trajectories.reshape(n, -1).astype(np.float32)


def run_knockoff_nets(target, train_windows: list, val_windows: list, seq_len: int, n_features: int, results: dict):
    print("\n" + "=" * 60)
    print("KNOCKOFF NETS -- real query-based model extraction")
    print("=" * 60)

    input_dim = seq_len * n_features
    query_func = pampos_target_query_func(target, seq_len, n_features)

    feature_mean = target.norm_mean.cpu().numpy()
    feature_std = target.norm_std.cpu().numpy()

    attack = KnockoffNetsAttack(input_dim=input_dim, output_dim=1, mode="classification", query_budget=500, strategy="adaptive")

    def sample_realistic(n):
        return sample_smooth_windows(n, seq_len, feature_mean, feature_std)

    attack._sample_candidates = sample_realistic

    x_eval = np.array([w.flatten() for w in val_windows[:100]]) if len(val_windows) >= 1 else sample_realistic(100)
    attack.extract(query_func, x_eval=x_eval, target_eval_func=query_func)

    print(f"Final fidelity (surrogate agreement with real PAMPOS): {attack.final_fidelity:.3f}")
    print(f"Total real PAMPOS queries used: {attack.query_count}")

    real_eval_labels = np.array([1 if query_func(x) >= 0.5 else 0 for x in x_eval])
    print(f"Real eval set label balance: {real_eval_labels.sum()}/{len(real_eval_labels)} labeled anomalous by PAMPOS")

    synthetic_probe = sample_realistic(200)
    synthetic_labels = np.array([1 if query_func(x) >= 0.5 else 0 for x in synthetic_probe])
    print(f"Synthetic query-candidate label balance: {synthetic_labels.sum()}/{len(synthetic_labels)} labeled anomalous by PAMPOS")

    n_anomalous = int(real_eval_labels.sum())
    if n_anomalous > 0:
        surrogate_preds = attack.predict(x_eval)
        surrogate_labels = (surrogate_preds[:, 0] >= 0.5).astype(int)
        anomalous_mask = real_eval_labels == 1
        recall_on_anomalous = float(np.mean(surrogate_labels[anomalous_mask] == 1))
        always_benign_fidelity = (len(real_eval_labels) - n_anomalous) / len(real_eval_labels)
        print(f"Recall on genuinely anomalous cases: {recall_on_anomalous:.3f} "
              f"(a naive always-benign guesser scores {always_benign_fidelity:.3f} fidelity for free -- "
              f"compare to the {attack.final_fidelity:.3f} reported above)")
        results["knockoff_nets_recall_on_anomalous"] = recall_on_anomalous
    else:
        print("[note] Zero genuinely anomalous windows in this eval sample -- the fidelity number above "
              "cannot be meaningfully interpreted as extraction success; it is inflated by class imbalance.")
        results["knockoff_nets_recall_on_anomalous"] = None

    results["knockoff_nets"] = attack.get_statistics()


def run_attribute_inference(target, train_windows: list, val_windows: list, seq_len: int, n_features: int, results: dict):
    print("\n" + "=" * 60)
    print("ATTRIBUTE INFERENCE BLACK BOX -- real is_camera_visible inference")
    print("=" * 60)

    input_dim = seq_len * n_features
    attribute_index = (seq_len - 1) * n_features + 6
    query_func = flatten_window_query_func(target, seq_len, n_features)

    aux_states = np.array([w.flatten() for w in train_windows[:300]])
    eval_states = np.array([w.flatten() for w in val_windows[:100]]) if len(val_windows) >= 10 else aux_states[:50]

    attack = AttributeInferenceBlackBoxAttack(input_dim=input_dim, attribute_index=attribute_index, attribute_values=[0.0, 1.0])
    attack.fit(aux_states, query_func)
    eval_results = attack.evaluate_accuracy(eval_states, query_func)

    print(f"Held-out accuracy: {eval_results['accuracy']:.3f} (baseline {eval_results['baseline_accuracy']:.3f})")
    print(f"Advantage over baseline: {eval_results['advantage_over_baseline']:+.3f}")

    results["attribute_inference"] = eval_results


def run_miface(target, train_windows: list, seq_len: int, n_features: int, results: dict):
    print("\n" + "=" * 60)
    print("MIFACE -- real model inversion")
    print("=" * 60)

    input_dim = seq_len * n_features
    input_bounds = compute_empirical_bounds(train_windows[:300], seq_len, n_features)
    query_func = flatten_window_query_func(target, seq_len, n_features)

    attack = MIFaceAttack(input_dim=input_dim, input_bounds=input_bounds, max_iterations=150, finite_diff_batch=15)
    attack.invert_all_classes(query_func)

    for target_class, result in attack.reconstructions.items():
        label = "ANOMALOUS" if target_class == 1 else "BENIGN"
        print(f"{label}: confidence={result['achieved_confidence']:.3f}, iterations={result['iterations_run']}")
        reconstructed = np.array(result["reconstructed_state"]).reshape(seq_len, n_features)
        last_step = dict(zip(FEATURE_NAMES, reconstructed[-1].tolist()))
        print(f"  Last timestep reconstructed values: {last_step}")
        at_bound_count = 0
        for i in range(input_dim):
            if abs(result["reconstructed_state"][i] - input_bounds[i, 0]) < 1e-3 or abs(result["reconstructed_state"][i] - input_bounds[i, 1]) < 1e-3:
                at_bound_count += 1
        print(f"  {at_bound_count}/{input_dim} values pinned at search bounds (high count suggests degenerate/unrealistic reconstruction)")
    print(f"Total gradient queries: {attack.gradient_queries}")

    results["miface"] = attack.get_statistics()


def run_hopskipjump(target, train_windows: list, seq_len: int, n_features: int, results: dict):
    print("\n" + "=" * 60)
    print("HOPSKIPJUMP -- real decision-boundary robustness")
    print("=" * 60)

    decision_func = make_pampos_decision_func(target, seq_len, n_features)

    benign_start = None
    for w in train_windows:
        flat = w.flatten()
        if decision_func(flat) == 0:
            benign_start = flat
            break

    if benign_start is None:
        print("[skip] Could not find a real window classified as benign under current calibration.")
        results["hopskipjump"] = {"skipped": True}
        return

    attack = HopSkipJumpAttack(max_iterations=15)
    delta = attack.attack(benign_start, decision_func)

    print(f"Minimal perturbation norm to flip real PAMPOS's decision: {np.linalg.norm(delta):.4f}")
    print(f"Real PAMPOS queries used: {attack.query_count}")

    results["hopskipjump"] = attack.get_statistics()


def run_database_reconstruction(target, train_windows: list, seq_len: int, n_features: int, results: dict):
    print("\n" + "=" * 60)
    print("DATABASE RECONSTRUCTION -- real feature data, PAMPOS-derived labels")
    print("=" * 60)

    input_dim = seq_len * n_features
    pool_size = min(2000, len(train_windows))
    pool_scores = [(i, target.raw_score(train_windows[i])) for i in range(pool_size)]
    flagged = [i for i, s in pool_scores if s > target.threshold]
    unflagged = [i for i, s in pool_scores if s <= target.threshold]

    n_each = min(30, len(flagged), len(unflagged))
    if n_each < 10:
        print(f"[warning] Only found {len(flagged)} flagged windows in a pool of {pool_size} -- "
              f"class imbalance limits sample size regardless of pool size. Using {n_each} per class "
              f"(reconstruction task will be data-starved with such a small sample).")
    if n_each == 0:
        print(f"[warning] Could not find both classes in first {pool_size} windows "
              f"({len(flagged)} flagged, {len(unflagged)} unflagged) -- falling back to whatever's available.")
        selected_indices = [i for i, _ in pool_scores[:60]]
    else:
        selected_indices = flagged[:n_each] + unflagged[:n_each]
        print(f"[setup] Balanced sample: {n_each} flagged + {n_each} unflagged real windows")

    X_full = np.array([train_windows[i].flatten() for i in selected_indices])
    y_full = np.array([1.0 if target.raw_score(train_windows[i]) > target.threshold else 0.0 for i in selected_indices])

    missing_idx = 0
    missing_row = X_full[missing_idx].copy()
    missing_label = y_full[missing_idx]
    known_X = np.delete(X_full, missing_idx, axis=0)
    known_y = np.delete(y_full, missing_idx, axis=0)

    surrogate_target = DatabaseTargetModel(input_dim=input_dim)
    surrogate_target.fit(X_full, y_full)

    attack = DatabaseReconstructionAttack(
        target_factory=lambda: DatabaseTargetModel(input_dim=input_dim),
        input_bounds=compute_empirical_bounds(train_windows[:300], seq_len, n_features),
        max_iterations=80,
    )
    result = attack.reconstruct(surrogate_target, known_X, known_y)

    feature_error = float(np.linalg.norm(np.array(result["reconstructed_features"]) - missing_row))
    print(f"Label recovered correctly: {result['candidate_label'] == missing_label}")
    print(f"Feature L2 error vs. true missing row: {feature_error:.3f}")
    print(f"Retrain queries used: {attack.gradient_queries}")

    results["database_reconstruction"] = {**attack.get_statistics(), "feature_l2_error": feature_error, "label_correct": bool(result["candidate_label"] == missing_label)}


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
    target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    print(f"[setup] Canonical threshold: {target.threshold:.4f}")

    print("[setup] Rebuilding real train/val split (identical to train_baseline.py)...")
    train_windows, val_windows = get_real_train_val_windows(config, REPO_ROOT)
    print(f"[setup] Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

    results = {}
    run_membership_inference(target, train_windows, val_windows, seq_len, n_features, results)
    run_knockoff_nets(target, train_windows, val_windows, seq_len, n_features, results)
    run_attribute_inference(target, train_windows, val_windows, seq_len, n_features, results)
    run_miface(target, train_windows, seq_len, n_features, results)
    run_hopskipjump(target, train_windows, seq_len, n_features, results)
    run_database_reconstruction(target, train_windows, seq_len, n_features, results)

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ml_attacks_real_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"[done] All results saved to {output_path}")


if __name__ == "__main__":
    main()