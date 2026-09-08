import json
import sys
from collections import defaultdict
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
from src.attacks.adversarial_ml_attacks.hopskipjump import HopSkipJumpAttack, make_pampos_decision_func

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]
N_SEEDS = 5
CALIBRATION_SIZE = 300
N_PROBES = 20


def robust_percentile_threshold(scores, percentile=99.0, trim_fraction=0.1):
    sorted_scores = np.sort(np.array(scores))
    n = len(sorted_scores)
    trim_n = int(n * trim_fraction)
    trimmed = sorted_scores[trim_n:n - trim_n] if trim_n > 0 else sorted_scores
    return float(np.percentile(trimmed, percentile))


def detect_calibration_outliers(windows, contamination_estimate=0.2):
    flat = np.concatenate([w.reshape(-1, w.shape[-1]) for w in windows], axis=0)
    median = np.median(flat, axis=0)
    mad = np.median(np.abs(flat - median), axis=0) + 1e-8
    robust_z = np.abs(flat - median) / (1.4826 * mad)
    score = robust_z.max(axis=1)
    thr = np.percentile(score, 100 * (1 - contamination_estimate))
    return score > thr, score


def clean_windows_via_outlier_detection(windows, contamination_estimate=0.2):
    seq_len = windows[0].shape[0]
    is_outlier, _ = detect_calibration_outliers(windows, contamination_estimate)
    per_window = is_outlier.reshape(len(windows), seq_len)
    cleaned = []
    for w, mask in zip(windows, per_window):
        if mask.any():
            clean_rows = w[~mask]
            if len(clean_rows) == 0:
                continue
            fill = clean_rows.mean(axis=0)
            wc = w.copy()
            wc[mask] = fill
            cleaned.append(wc)
        else:
            cleaned.append(w)
    return cleaned


def clean_windows_via_local_consistency(windows, z_threshold=3.0):
    cleaned = []
    for w in windows:
        wc = w.copy()
        for t in range(w.shape[0]):
            others = np.delete(w, t, axis=0)
            med = np.median(others, axis=0)
            mad = np.median(np.abs(others - med), axis=0) + 1e-8
            z = np.abs(w[t] - med) / (1.4826 * mad)
            if z.max() > z_threshold:
                wc[t] = med
        cleaned.append(wc)
    return cleaned


def run_backdoor_trial(target, calibration_windows, all_windows, seed):
    target.feature_mae = None
    target.calibrate(calibration_windows)
    clean_threshold = target.threshold
    clean_feature_mae = target.feature_mae.clone()

    scores = [target.raw_score(w) for w in all_windows]
    flagged = [(w, s) for w, s in zip(all_windows, scores) if s > clean_threshold]
    flagged.sort(key=lambda p: p[1], reverse=True)
    k = min(N_PROBES, len(flagged))
    probes = [w for w, s in flagged[:k]]

    if k == 0:
        return None

    def score_with(feature_mae, threshold, window):
        target.feature_mae = feature_mae
        target.threshold = threshold
        return target.raw_score(window)

    results = {"n_flagged": len(flagged), "n_probes": k, "clean_threshold": clean_threshold}

    hidden_naive = 0
    hidden_trimmed = 0
    hidden_outlier = 0
    hidden_local = 0

    for probe in probes:
        trigger = probe.mean(axis=0)
        attack = create_backdoor_attack(
            trigger_pattern=dict(zip(FEATURE_NAMES, trigger.tolist())),
            fraction_poisoned=0.15,
        )
        poisoned = attack.poison(calibration_windows, seed=seed)

        target.feature_mae = None
        target.calibrate(poisoned)
        if target.raw_score(probe) <= target.threshold:
            hidden_naive += 1

        target.feature_mae = None
        target.calibrate(poisoned)
        poisoned_scores = [target.raw_score(w) for w in poisoned]
        trimmed_thr = robust_percentile_threshold(poisoned_scores)
        if target.raw_score(probe) <= trimmed_thr:
            hidden_trimmed += 1

        cleaned_out = clean_windows_via_outlier_detection(poisoned)
        target.feature_mae = None
        target.calibrate(cleaned_out)
        if target.raw_score(probe) <= target.threshold:
            hidden_outlier += 1

        cleaned_loc = clean_windows_via_local_consistency(poisoned)
        target.feature_mae = None
        target.calibrate(cleaned_loc)
        if target.raw_score(probe) <= target.threshold:
            hidden_local += 1

    results.update({
        "naive": hidden_naive, "trimmed": hidden_trimmed,
        "outlier": hidden_outlier, "local": hidden_local,
    })

    target.feature_mae = clean_feature_mae
    target.threshold = clean_threshold
    return results


def run_hopskipjump_trials(target, train_windows, seq_len, n_features, n_trials=5):
    decision_func = make_pampos_decision_func(target, seq_len, n_features)
    norms = []

    benign_candidates = []
    for w in train_windows:
        flat = w.flatten()
        if decision_func(flat) == 0:
            benign_candidates.append(flat)
        if len(benign_candidates) >= n_trials:
            break

    for start in benign_candidates:
        attack = HopSkipJumpAttack(max_iterations=15)
        try:
            delta = attack.attack(start, decision_func)
            norms.append(float(np.linalg.norm(delta)))
        except Exception as e:
            print(f"    [skip] trial failed: {e}")

    return norms


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    n_features = config["model"]["input_dim"]
    model_config = {
        "input_dim": n_features, "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint...")
    target = load_pampos_target(REPO_ROOT, model_config)

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    gen = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, _ = random_split(full_dataset, [train_size, val_size], generator=gen)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    all_windows = [full_dataset.sequences[i] for i in range(len(full_dataset))]

    print(f"[setup] {len(train_windows)} train windows, {len(all_windows)} total windows")
    print(f"[setup] Running {N_SEEDS} independent trials with different calibration subsets\n")

    print("=" * 78)
    print("BACKDOOR DEFENSE -- MULTI-SEED VARIANCE")
    print("=" * 78)
    print(f"{'Seed':<8}{'Flagged':<10}{'Probes':<9}{'Thresh':<10}{'Naive':<9}{'Trimmed':<10}{'Outlier':<10}{'Local':<8}")
    print("-" * 78)

    trials = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_windows), size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
        calibration_windows = [train_windows[i] for i in idx]

        result = run_backdoor_trial(target, calibration_windows, all_windows, seed)
        if result is None:
            print(f"{seed:<8}no flagged windows -- skipped")
            continue

        trials.append(result)
        print(f"{seed:<8}{result['n_flagged']:<10}{result['n_probes']:<9}"
              f"{result['clean_threshold']:<10.3f}{result['naive']:<9}{result['trimmed']:<10}"
              f"{result['outlier']:<10}{result['local']:<8}")

    if trials:
        print("-" * 78)
        print("\nSUMMARY (mean +/- std across seeds, as fraction of probes hidden)")
        print("-" * 78)
        for key, label in [("naive", "No defense"), ("trimmed", "Trimmed-percentile"),
                           ("outlier", "Outlier-detection"), ("local", "Local consistency")]:
            fracs = [t[key] / t["n_probes"] for t in trials]
            counts = [t[key] for t in trials]
            print(f"{label:<24}{np.mean(fracs)*100:>6.1f}% +/- {np.std(fracs)*100:>5.1f}%   "
                  f"(raw counts: {counts})")

        thresholds = [t["clean_threshold"] for t in trials]
        flagged_counts = [t["n_flagged"] for t in trials]
        print(f"\n{'Clean threshold':<24}{np.mean(thresholds):>6.3f} +/- {np.std(thresholds):>5.3f}   {[round(t,2) for t in thresholds]}")
        print(f"{'Flagged windows found':<24}{np.mean(flagged_counts):>6.1f} +/- {np.std(flagged_counts):>5.1f}   {flagged_counts}")

    print("\n" + "=" * 78)
    print("HOPSKIPJUMP -- MULTI-TRIAL VARIANCE")
    print("=" * 78)

    target.feature_mae = None
    target.calibrate(train_windows[:CALIBRATION_SIZE])
    norms = run_hopskipjump_trials(target, train_windows, seq_len, n_features, n_trials=5)
    if norms:
        print(f"Perturbation norms across {len(norms)} trials: {[round(n, 4) for n in norms]}")
        print(f"Mean +/- std: {np.mean(norms):.4f} +/- {np.std(norms):.4f}")
        print(f"Min / Max:    {min(norms):.4f} / {max(norms):.4f}")
    else:
        print("No successful trials.")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "variance_analysis.json", "w") as f:
        json.dump({
            "n_seeds": N_SEEDS,
            "backdoor_trials": trials,
            "hopskipjump_norms": norms,
        }, f, indent=2, default=str)
    print(f"\n[done] Saved to outputs/results/variance_analysis.json")


if __name__ == "__main__":
    main()