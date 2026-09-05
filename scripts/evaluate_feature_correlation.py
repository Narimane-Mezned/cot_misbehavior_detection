import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target
from src.attacks.adversarial_ml_attacks.attribute_inference_black_box import (
    AttributeInferenceBlackBoxAttack, flatten_window_query_func,
)

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "distance_to_ego"]
OTHER_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 7]
ATTRIBUTE_INDEX_IN_ROW = 6


def naive_correlation_classifier(train_windows: list, val_windows: list) -> dict:
    def extract_last_step(windows):
        X, y = [], []
        for w in windows:
            last = w[-1]
            X.append(last[OTHER_FEATURE_INDICES])
            y.append(last[ATTRIBUTE_INDEX_IN_ROW])
        return np.array(X), np.array(y)

    X_train, y_train = extract_last_step(train_windows)
    X_val, y_val = extract_last_step(val_windows)

    baseline_accuracy = max(y_val.mean(), 1 - y_val.mean())

    correlations = {}
    for i, name in enumerate(FEATURE_NAMES):
        if X_train[:, i].std() > 1e-8:
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlations[name] = float(corr)

    best_idx = int(np.argmax([abs(v) for v in correlations.values()]))
    best_name = list(correlations.keys())[best_idx]
    best_train = X_train[:, best_idx]
    best_val = X_val[:, best_idx]

    thresholds = np.percentile(best_train, np.arange(1, 100))
    best_acc, best_threshold, best_direction = 0.0, None, "above"
    for t in thresholds:
        for direction in ["above", "below"]:
            preds = (best_train > t).astype(float) if direction == "above" else (best_train < t).astype(float)
            acc = np.mean(preds == y_train)
            if acc > best_acc:
                best_acc, best_threshold, best_direction = acc, t, direction

    val_preds = (best_val > best_threshold).astype(float) if best_direction == "above" else (best_val < best_threshold).astype(float)
    val_acc = float(np.mean(val_preds == y_val))

    return {
        "correlations": correlations,
        "strongest_feature": best_name,
        "naive_val_accuracy": val_acc,
        "true_baseline_accuracy": float(baseline_accuracy),
        "n_train": len(y_train), "n_val": len(y_val),
    }


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

    print("[setup] Rebuilding the EXACT SAME real train/val split used everywhere else...")
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    val_windows = [full_dataset.sequences[i] for i in val_subset.indices]
    print(f"[setup] Train: {len(train_windows)}, Val: {len(val_windows)}")

    aux_windows = train_windows[:300]
    eval_windows = val_windows[:100]

    print("\n[test 1/2] Naive correlation classifier -- ZERO PAMPOS queries, same exact windows")
    naive_results = naive_correlation_classifier(aux_windows, eval_windows)
    print(f"  Correlations: {naive_results['correlations']}")
    print(f"  Strongest feature: {naive_results['strongest_feature']}")
    print(f"  Naive classifier held-out accuracy: {naive_results['naive_val_accuracy']:.3f}")
    print(f"  TRUE majority-class baseline (on these {naive_results['n_val']} windows): {naive_results['true_baseline_accuracy']:.3f}")

    print("\n[setup] Loading real PAMPOS checkpoint for direct re-comparison...")
    target = load_pampos_target(REPO_ROOT, model_config)
    target.calibrate(train_windows[:300])
    print(f"[setup] Threshold: {target.threshold:.4f}")

    print("\n[test 2/2] Attribute Inference attack -- DOES query PAMPOS, same exact windows")
    attribute_index = (seq_len - 1) * n_features + 6
    query_func = flatten_window_query_func(target, seq_len, n_features)
    aux_states = np.array([w.flatten() for w in aux_windows])
    eval_states = np.array([w.flatten() for w in eval_windows])

    attack = AttributeInferenceBlackBoxAttack(input_dim=seq_len * n_features, attribute_index=attribute_index, attribute_values=[0.0, 1.0])
    attack.fit(aux_states, query_func)
    attack_results = attack.evaluate_accuracy(eval_states, query_func)
    print(f"  Attack accuracy: {attack_results['accuracy']:.3f}")
    print(f"  Attack's own reported baseline: {attack_results['baseline_accuracy']:.3f}")
    print(f"  Attack's own reported advantage: {attack_results['advantage_over_baseline']:+.3f}")

    print("\n" + "=" * 75)
    print("FINAL, RECONCILED COMPARISON (identical windows used for both tests)")
    print("=" * 75)
    print(f"  True majority-class baseline:              {naive_results['true_baseline_accuracy']:.3f}")
    print(f"  Naive classifier (NO model queries):        {naive_results['naive_val_accuracy']:.3f}")
    print(f"  Attribute Inference (DOES query PAMPOS):    {attack_results['accuracy']:.3f}")
    real_model_specific_advantage = attack_results['accuracy'] - naive_results['naive_val_accuracy']
    print(f"\n  Real model-specific advantage (attack accuracy minus naive-correlation accuracy): {real_model_specific_advantage:+.3f}")
    print("  This is the honest number: how much MORE does querying PAMPOS reveal, beyond")
    print("  what's already predictable from natural sensor physics alone.")


if __name__ == "__main__":
    main()