import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset


def naive_correlation_classifier(train_windows: list, val_windows: list, seq_len: int) -> dict:
    attribute_idx = 6
    other_feature_indices = [0, 1, 2, 3, 4, 5, 7]

    def extract_last_step(windows):
        X, y = [], []
        for w in windows:
            last = w[-1]
            X.append(last[other_feature_indices])
            y.append(last[attribute_idx])
        return np.array(X), np.array(y)

    X_train, y_train = extract_last_step(train_windows[:300])
    X_val, y_val = extract_last_step(val_windows[:100])

    baseline_accuracy = max(y_val.mean(), 1 - y_val.mean())

    correlations = {}
    feature_names = ["x", "y", "vx", "vy", "yaw", "point_count", "distance_to_ego"]
    for i, name in enumerate(feature_names):
        if X_train[:, i].std() > 1e-8:
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlations[name] = float(corr)

    best_feature_idx = int(np.argmax([abs(v) for v in correlations.values()]))
    best_feature_name = list(correlations.keys())[best_feature_idx]
    best_feature_train = X_train[:, best_feature_idx]
    best_feature_val = X_val[:, best_feature_idx]

    thresholds = np.percentile(best_feature_train, np.arange(1, 100))
    best_acc, best_threshold, best_direction = 0.0, None, "above"
    for t in thresholds:
        for direction in ["above", "below"]:
            preds = (best_feature_train > t).astype(float) if direction == "above" else (best_feature_train < t).astype(float)
            acc = np.mean(preds == y_train)
            if acc > best_acc:
                best_acc, best_threshold, best_direction = acc, t, direction

    val_preds = (best_feature_val > best_threshold).astype(float) if best_direction == "above" else (best_feature_val < best_threshold).astype(float)
    val_acc = np.mean(val_preds == y_val)

    return {
        "correlations_with_is_camera_visible": correlations,
        "strongest_single_feature": best_feature_name,
        "naive_threshold_train_accuracy": float(best_acc),
        "naive_threshold_val_accuracy": float(val_acc),
        "baseline_majority_class_accuracy": float(baseline_accuracy),
    }


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    data_root = REPO_ROOT / config["data"]["raw_dir"]

    print("[setup] Loading real DeepAccident windows (no PAMPOS involved at all)...")
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    windows = [full_dataset.sequences[i] for i in range(len(full_dataset))]

    split = int(len(windows) * 0.85)
    train_windows = windows[:split]
    val_windows = windows[split:]

    print(f"[setup] {len(train_windows)} train windows, {len(val_windows)} val windows")
    print("\n[test] Can is_camera_visible be predicted from the OTHER 7 real features, with ZERO model queries?\n")

    results = naive_correlation_classifier(train_windows, val_windows, seq_len)

    print("Correlation of each other feature with is_camera_visible:")
    for name, corr in results["correlations_with_is_camera_visible"].items():
        print(f"  {name:>15}: {corr:+.3f}")

    print(f"\nStrongest single feature: {results['strongest_single_feature']}")
    print(f"Naive threshold classifier (train): {results['naive_threshold_train_accuracy']:.3f}")
    print(f"Naive threshold classifier (held-out val): {results['naive_threshold_val_accuracy']:.3f}")
    print(f"Majority-class baseline: {results['baseline_majority_class_accuracy']:.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("Compare 'naive threshold classifier (held-out val)' above to Attribute")
    print("Inference's real result of 0.950 accuracy (which DOES query PAMPOS).")
    print("If the naive, model-free number is close to 0.950, the 'leak' is really")
    print("just natural sensor physics (distance/point_count correlate with")
    print("visibility) and has little to do with PAMPOS specifically.")
    print("If the naive number is much lower (e.g. near the 0.620 baseline),")
    print("PAMPOS's output is providing real additional information beyond what's")
    print("naturally in the data -- confirming a genuine model-specific leak.")


if __name__ == "__main__":
    main()