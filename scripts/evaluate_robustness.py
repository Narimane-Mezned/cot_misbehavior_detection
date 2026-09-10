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
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target
from src.attacks.adversarial_ml_attacks.backdoor_attack import create_backdoor_attack

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]
DISTANCE_IDX = 7
LIDAR_RANGE_M = 80.0
PERCENTILES = [95.0, 97.0, 99.0]
N_PROBES = 20
CALIBRATION_SIZE = 300
Z_THRESHOLD = 1.5


def clean_local(windows, z=Z_THRESHOLD):
    out = []
    for w in windows:
        wc = w.copy()
        for t in range(w.shape[0]):
            others = np.delete(w, t, axis=0)
            med = np.median(others, axis=0)
            mad = np.median(np.abs(others - med), axis=0) + 1e-8
            if (np.abs(w[t] - med) / (1.4826 * mad)).max() > z:
                wc[t] = med
        out.append(wc)
    return out


def calibrate_at(target, windows, percentile):
    target.feature_mae = None
    target.calibrate(windows, percentile=percentile)
    return target.threshold


def threshold_sensitivity(target, calibration_windows, all_windows):
    print("=" * 82)
    print("THRESHOLD SENSITIVITY -- do the findings survive a different cutoff?")
    print("=" * 82)
    print(f"{'percentile':<14}{'threshold':<14}{'flagged':<14}{'backdoor undef.':<20}{'backdoor defended':<18}")
    print("-" * 82)

    results = {}
    for pct in PERCENTILES:
        thr = calibrate_at(target, calibration_windows, pct)
        feature_mae = target.feature_mae.clone()

        scores = [target.raw_score(w) for w in all_windows]
        flagged = [(w, s) for w, s in zip(all_windows, scores) if s > thr]
        flagged.sort(key=lambda p: p[1], reverse=True)
        k = min(N_PROBES, len(flagged))

        hid_u = hid_d = 0
        for probe, _ in flagged[:k]:
            trig = probe.mean(axis=0)
            atk = create_backdoor_attack(
                trigger_pattern=dict(zip(FEATURE_NAMES, trig.tolist())),
                fraction_poisoned=0.15)
            poisoned = atk.poison(calibration_windows, seed=0)

            target.feature_mae = None
            target.calibrate(poisoned, percentile=pct)
            if target.raw_score(probe) <= target.threshold:
                hid_u += 1

            target.feature_mae = None
            target.calibrate(clean_local(poisoned), percentile=pct)
            if target.raw_score(probe) <= target.threshold:
                hid_d += 1

        ru = f"{100*hid_u/k:.0f}% ({hid_u}/{k})" if k else "n/a"
        rd = f"{100*hid_d/k:.0f}% ({hid_d}/{k})" if k else "n/a"
        print(f"{pct:<14.0f}{thr:<14.4f}{len(flagged):<14}{ru:<20}{rd:<18}")

        results[str(pct)] = {
            "threshold": thr, "n_flagged": len(flagged), "n_probes": k,
            "backdoor_undefended": hid_u, "backdoor_defended": hid_d,
        }

        target.feature_mae = feature_mae
        target.threshold = thr

    print("-" * 82)
    print("If attack success and defence effectiveness are stable across percentiles, the")
    print("findings are not an artifact of the 99th-percentile choice.\n")
    return results


def range_stratified(target, all_windows, threshold):
    print("=" * 82)
    print("RANGE-STRATIFIED ANALYSIS -- where does sensor grounding actually contribute?")
    print("=" * 82)

    mean_dist = np.array([w[:, DISTANCE_IDX].mean() for w in all_windows])
    near = mean_dist <= LIDAR_RANGE_M
    far = ~near

    print(f"near (<= {LIDAR_RANGE_M:.0f}m): {near.sum()} windows")
    print(f"far  (>  {LIDAR_RANGE_M:.0f}m): {far.sum()} windows "
          f"({100*far.sum()/len(all_windows):.1f}% of the dataset)\n")

    errs, scores = [], []
    for w in all_windows:
        s, e = target.score_with_breakdown(w)
        errs.append(e)
        scores.append(s)
    errs = np.array(errs)
    scores = np.array(scores)

    print(f"{'group':<10}{'n':<9}{'flagged':<12}{'flagged rate':<16}{'mean score':<12}")
    print("-" * 82)
    out = {}
    for label, mask in [("near", near), ("far", far)]:
        n = int(mask.sum())
        f = int((scores[mask] > threshold).sum())
        print(f"{label:<10}{n:<9}{f:<12}{100*f/n if n else 0:<16.2f}{scores[mask].mean():<12.3f}")
        out[label] = {"n": n, "flagged": f, "flagged_rate": f/n if n else None,
                      "mean_score": float(scores[mask].mean())}

    print()
    print("Per-feature discrimination (mean error on flagged / mean error on unflagged):")
    print(f"{'feature':<20}{'near':<14}{'far':<14}")
    print("-" * 82)
    for i, name in enumerate(FEATURE_NAMES):
        row = []
        for mask in [near, far]:
            fl = scores[mask] > threshold
            sub = errs[mask]
            if fl.sum() == 0 or (~fl).sum() == 0:
                row.append(None)
            else:
                a = sub[~fl, i].mean()
                b = sub[fl, i].mean()
                row.append(b/a if a > 0 else None)
        fmt = lambda v: f"{v:.2f}x" if v is not None else "n/a"
        print(f"{name:<20}{fmt(row[0]):<14}{fmt(row[1]):<14}")
        out.setdefault("discrimination", {})[name] = {
            "near": row[0], "far": row[1]}

    print()
    print("point_count is zero by physics beyond LiDAR range, so a low 'far' value there")
    print("would confirm that sensor grounding contributes nothing for distant agents.")
    return out


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] loading checkpoint...")
    target = load_pampos_target(REPO_ROOT, model_config)

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    ds = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    gen = torch.Generator().manual_seed(config["training"]["seed"])
    val_size = max(1, int(len(ds) * config["training"]["val_fraction"]))
    tr_sub, _ = random_split(ds, [len(ds) - val_size, val_size], generator=gen)
    train_windows = [ds.sequences[i] for i in tr_sub.indices]
    all_windows = [ds.sequences[i] for i in range(len(ds))]
    calibration_windows = train_windows[:CALIBRATION_SIZE]
    print(f"[setup] {len(all_windows)} windows total, {len(calibration_windows)} for calibration\n")

    ts = threshold_sensitivity(target, calibration_windows, all_windows)

    thr99 = calibrate_at(target, calibration_windows, 99.0)
    rs = range_stratified(target, all_windows, thr99)

    out_dir = REPO_ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "robustness_analysis.json", "w") as f:
        json.dump({"threshold_sensitivity": ts, "range_stratified": rs,
                   "lidar_range_m": LIDAR_RANGE_M}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/robustness_analysis.json")


if __name__ == "__main__":
    main()