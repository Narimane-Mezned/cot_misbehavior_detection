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

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    windows = [dataset.sequences[i] for i in range(len(dataset))]

    print(f"[setup] {len(windows)} windows, threshold {target.threshold:.4f}\n")

    print("=" * 78)
    print("1. RAW FEATURE SCALES (before normalization)")
    print("=" * 78)
    stacked = np.concatenate([w for w in windows], axis=0)
    print(f"{'feature':<22}{'min':>12}{'max':>12}{'mean':>12}{'std':>12}")
    for i, name in enumerate(FEATURE_NAMES):
        col = stacked[:, i]
        print(f"{name:<22}{col.min():>12.2f}{col.max():>12.2f}{col.mean():>12.2f}{col.std():>12.2f}")

    print()
    print("=" * 78)
    print("2. feature_mae (the divisor used to normalize prediction errors)")
    print("=" * 78)
    mae = target.feature_mae.cpu().numpy()
    print(f"{'feature':<22}{'feature_mae':>14}")
    for name, m in zip(FEATURE_NAMES, mae):
        print(f"{name:<22}{m:>14.6f}")
    print("\nA SMALL feature_mae divides errors by a small number, inflating that feature's")
    print("normalized error and making it dominate the top-K breakdown.")

    print()
    print("=" * 78)
    print("3. HOW OFTEN EACH FEATURE IS THE TOP CONTRIBUTOR (all windows)")
    print("=" * 78)
    top_counts = {n: 0 for n in FEATURE_NAMES}
    all_errs = []
    for w in windows:
        _, errs = target.score_with_breakdown(w)
        all_errs.append(errs)
        top_counts[FEATURE_NAMES[int(np.argmax(errs))]] += 1

    all_errs = np.array(all_errs)
    total = len(windows)
    for name, c in sorted(top_counts.items(), key=lambda p: -p[1]):
        print(f"   {name:<22}{c:>6} / {total}  ({100*c/total:>5.1f}%)")

    print()
    print("=" * 78)
    print("4. MEAN NORMALIZED ERROR PER FEATURE")
    print("=" * 78)
    print(f"{'feature':<22}{'mean':>12}{'median':>12}{'p99':>12}{'max':>12}")
    for i, name in enumerate(FEATURE_NAMES):
        col = all_errs[:, i]
        print(f"{name:<22}{col.mean():>12.2f}{np.median(col):>12.2f}"
              f"{np.percentile(col,99):>12.2f}{col.max():>12.2f}")

    print()
    print("=" * 78)
    print("5. DOES point_count ACTUALLY DISCRIMINATE FLAGGED FROM UNFLAGGED?")
    print("=" * 78)
    scores = np.array([target.raw_score(w) for w in windows])
    flagged = scores > target.threshold
    print(f"flagged: {flagged.sum()} / {len(windows)}\n")
    print(f"{'feature':<22}{'mean(unflagged)':>18}{'mean(flagged)':>16}{'ratio':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        a = all_errs[~flagged, i].mean()
        b = all_errs[flagged, i].mean()
        ratio = b / a if a > 0 else float("inf")
        print(f"{name:<22}{a:>18.2f}{b:>16.2f}{ratio:>10.2f}x")

    print("\nA HIGH ratio means the feature genuinely separates anomalous from normal windows.")
    print("A ratio near 1.0 means it is noisy in both cases and dominance is a scaling artifact.")


if __name__ == "__main__":
    main()