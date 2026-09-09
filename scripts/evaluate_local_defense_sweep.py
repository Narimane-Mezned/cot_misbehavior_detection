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
Z_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
N_SEEDS = 3
CALIBRATION_SIZE = 300
N_PROBES = 20


def clean_windows_via_local_consistency(windows, z_threshold):
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


def count_replaced_timesteps(windows, z_threshold):
    replaced = 0
    total = 0
    for w in windows:
        for t in range(w.shape[0]):
            total += 1
            others = np.delete(w, t, axis=0)
            med = np.median(others, axis=0)
            mad = np.median(np.abs(others - med), axis=0) + 1e-8
            z = np.abs(w[t] - med) / (1.4826 * mad)
            if z.max() > z_threshold:
                replaced += 1
    return replaced, total


def run_seed_trial(target, train_windows, all_windows, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
    calibration_windows = [train_windows[i] for i in idx]

    target.feature_mae = None
    target.calibrate(calibration_windows)
    clean_threshold = target.threshold

    scores = [target.raw_score(w) for w in all_windows]
    flagged = [(w, s) for w, s in zip(all_windows, scores) if s > clean_threshold]
    flagged.sort(key=lambda p: p[1], reverse=True)
    k = min(N_PROBES, len(flagged))
    if k == 0:
        return None
    probes = [w for w, s in flagged[:k]]

    results = {"seed": seed, "clean_threshold": clean_threshold, "n_flagged": len(flagged),
               "n_probes": k, "by_z": {}}

    poisoned_sets = []
    for probe in probes:
        trigger = probe.mean(axis=0)
        attack = create_backdoor_attack(
            trigger_pattern=dict(zip(FEATURE_NAMES, trigger.tolist())),
            fraction_poisoned=0.15,
        )
        poisoned_sets.append(attack.poison(calibration_windows, seed=seed))

    for z in Z_THRESHOLDS:
        hidden = 0
        for probe, poisoned in zip(probes, poisoned_sets):
            cleaned = clean_windows_via_local_consistency(poisoned, z)
            target.feature_mae = None
            target.calibrate(cleaned)
            if target.raw_score(probe) <= target.threshold:
                hidden += 1

        cleaned_clean = clean_windows_via_local_consistency(calibration_windows, z)
        target.feature_mae = None
        target.calibrate(cleaned_clean)
        cleaned_threshold = target.threshold

        still_flagged = sum(1 for probe in probes if target.raw_score(probe) > cleaned_threshold)
        replaced, total = count_replaced_timesteps(calibration_windows, z)

        results["by_z"][str(z)] = {
            "hidden": hidden,
            "hidden_frac": hidden / k,
            "utility_still_flagged": still_flagged,
            "utility_frac": still_flagged / k,
            "cleaned_threshold": cleaned_threshold,
            "timesteps_replaced_frac": replaced / total if total else 0.0,
        }

    target.feature_mae = None
    target.calibrate(calibration_windows)
    return results


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

    print(f"[setup] Sweeping z_threshold over {Z_THRESHOLDS}")
    print(f"[setup] {N_SEEDS} seeds x {len(Z_THRESHOLDS)} settings x {N_PROBES} probes")
    print(f"[setup] This is heavy -- expect a long run.\n")

    trials = []
    for seed in range(N_SEEDS):
        print(f"[seed {seed}] running...")
        result = run_seed_trial(target, train_windows, all_windows, seed)
        if result is None:
            print(f"[seed {seed}] no flagged windows, skipped")
            continue
        trials.append(result)
        print(f"[seed {seed}] threshold={result['clean_threshold']:.3f}, "
              f"flagged={result['n_flagged']}, probes={result['n_probes']}")
        for z in Z_THRESHOLDS:
            r = result["by_z"][str(z)]
            print(f"    z={z}: attack hides {r['hidden']}/{result['n_probes']} "
                  f"| real anomalies still caught {r['utility_still_flagged']}/{result['n_probes']} "
                  f"| timesteps replaced {100*r['timesteps_replaced_frac']:.1f}%")
        print()

    if not trials:
        print("[error] No usable trials.")
        return

    print("=" * 88)
    print("SWEEP SUMMARY (mean +/- std across seeds)")
    print("=" * 88)
    print(f"{'z':<8}{'Attack success':<22}{'Real anomalies kept':<24}{'Timesteps replaced':<20}")
    print("-" * 88)

    summary = {}
    for z in Z_THRESHOLDS:
        hidden = [t["by_z"][str(z)]["hidden_frac"] for t in trials]
        utility = [t["by_z"][str(z)]["utility_frac"] for t in trials]
        replaced = [t["by_z"][str(z)]["timesteps_replaced_frac"] for t in trials]

        print(f"{z:<8}{np.mean(hidden)*100:>6.1f}% +/- {np.std(hidden)*100:<10.1f}"
              f"{np.mean(utility)*100:>6.1f}% +/- {np.std(utility)*100:<12.1f}"
              f"{np.mean(replaced)*100:>6.1f}%")

        summary[str(z)] = {
            "attack_success_mean": float(np.mean(hidden)),
            "attack_success_std": float(np.std(hidden)),
            "utility_kept_mean": float(np.mean(utility)),
            "utility_kept_std": float(np.std(utility)),
            "timesteps_replaced_mean": float(np.mean(replaced)),
        }

    print("-" * 88)
    print("\nBaseline for comparison: with NO defense, attack success is 100%.")
    print("'Real anomalies kept' = fraction of genuinely anomalous windows still correctly")
    print("flagged after cleaning CLEAN (unpoisoned) calibration data -- this is the utility cost.")

    print("\n" + "=" * 88)
    print("BEST TRADEOFF")
    print("=" * 88)
    viable = [(z, s) for z, s in summary.items() if s["utility_kept_mean"] >= 0.9]
    if viable:
        best_z, best = min(viable, key=lambda p: p[1]["attack_success_mean"])
        print(f"At z={best_z}: attack success {best['attack_success_mean']*100:.1f}% "
              f"(+/- {best['attack_success_std']*100:.1f}%), "
              f"real anomalies kept {best['utility_kept_mean']*100:.1f}%")
        print(f"That is a reduction from 100% (no defense) to {best['attack_success_mean']*100:.1f}%, "
              f"at a utility cost of {(1-best['utility_kept_mean'])*100:.1f}%.")
    else:
        print("No setting kept >=90% of real anomalies. The defense cannot be tuned to be")
        print("both effective and low-cost on this data -- report the full tradeoff curve honestly.")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "local_defense_sweep.json", "w") as f:
        json.dump({"z_thresholds": Z_THRESHOLDS, "n_seeds": len(trials),
                   "summary": summary, "trials": trials}, f, indent=2, default=str)
    print(f"\n[done] Saved to outputs/results/local_defense_sweep.json")


if __name__ == "__main__":
    main()