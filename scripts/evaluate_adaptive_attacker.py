import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import PAMPOSTarget

Z_THRESHOLD = 1.5
CALIBRATION_SIZE = 300
N_PROBES = 20
SEEDS = [0, 1, 2]

ALPHAS = [0.5, 0.7, 0.9, 1.1, 1.5, 3.0]
FRACTIONS = [0.15, 0.30, 0.50]


def filter_windows(windows, z_threshold=Z_THRESHOLD):
    cleaned, removed, total = [], 0, 0
    for w in windows:
        wc = w.copy()
        for t in range(w.shape[0]):
            others = np.delete(w, t, axis=0)
            med = np.median(others, axis=0)
            mad = np.median(np.abs(others - med), axis=0) + 1e-8
            z = np.abs(w[t] - med) / (1.4826 * mad)
            total += 1
            if z.max() > z_threshold:
                wc[t] = med
                removed += 1
        cleaned.append(wc)
    return cleaned, (removed / total if total else 0.0)


def z_of_point(window, t, candidate):
    others = np.delete(window, t, axis=0)
    med = np.median(others, axis=0)
    mad = np.median(np.abs(others - med), axis=0) + 1e-8
    return float(np.max(np.abs(candidate - med) / (1.4826 * mad)))


def poison_adaptive(windows, trigger, alpha, fraction, rng):
    """Move poisoned points toward the trigger, but only as far as a z-score of
    alpha * Z_THRESHOLD. alpha < 1 survives the filter; alpha > 1 is removed."""
    out = [w.copy() for w in windows]
    n_points = len(windows) * windows[0].shape[0]
    k = int(n_points * fraction)
    locs = rng.choice(n_points, size=k, replace=False)

    for loc in locs:
        wi, t = divmod(loc, windows[0].shape[0])
        w = out[wi]
        others = np.delete(w, t, axis=0)
        med = np.median(others, axis=0)
        mad = np.median(np.abs(others - med), axis=0) + 1e-8

        direction = trigger - med
        norm_z = np.abs(direction) / (1.4826 * mad)
        peak = float(norm_z.max())
        if peak < 1e-9:
            continue
        scale = min(1.0, (alpha * Z_THRESHOLD) / peak)
        w[t] = med + direction * scale
    return out


def build_trigger_from_distribution(benign_pool, rng):
    """Weakest adversary: knows only the benign distribution."""
    flat = np.concatenate([w for w in benign_pool], axis=0)
    lo = np.percentile(flat, 1, axis=0)
    hi = np.percentile(flat, 99, axis=0)
    return lo + rng.random(flat.shape[1]) * (hi - lo)


def run_seed(target, train_windows, all_windows, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows),
                     size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
    calib = [train_windows[i] for i in idx]

    target.feature_mae = None
    target.calibrate(calib)
    clean_threshold = target.threshold

    scores = [target.raw_score(w) for w in all_windows]
    flagged = sorted([(w, s) for w, s in zip(all_windows, scores) if s > clean_threshold],
                     key=lambda p: p[1], reverse=True)
    if not flagged:
        return None
    probes = [w for w, _ in flagged[:N_PROBES]]

    benign_scores = [target.raw_score(w) for w in calib]

    out = {"seed": seed, "clean_threshold": clean_threshold, "cells": {}}

    for alpha in ALPHAS:
        for frac in FRACTIONS:
            hidden, fa_rates, removal_rates = 0, [], []
            for probe in probes:
                trigger = build_trigger_from_distribution(calib, rng)
                poisoned = poison_adaptive(calib, trigger, alpha, frac, rng)
                cleaned, removal = filter_windows(poisoned)
                removal_rates.append(removal)

                target.feature_mae = None
                target.calibrate(cleaned)
                if target.raw_score(probe) <= target.threshold:
                    hidden += 1
                fa = np.mean([s > target.threshold for s in benign_scores])
                fa_rates.append(float(fa))

            out["cells"][f"{alpha}_{frac}"] = {
                "alpha": alpha, "fraction": frac,
                "attack_success": 100.0 * hidden / len(probes),
                "filter_removal": 100.0 * float(np.mean(removal_rates)),
                "false_alarm": 100.0 * float(np.mean(fa_rates)),
            }

    target.feature_mae = None
    target.calibrate(calib)
    return out


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] device {device}")
    print(f"[setup] filter z-threshold {Z_THRESHOLD}, calibration {CALIBRATION_SIZE} windows")
    print(f"[setup] the adaptive attacker scales its injection to a z-score of")
    print(f"[setup] alpha * {Z_THRESHOLD}. alpha below 1 evades the filter by")
    print(f"[setup] construction; alpha above 1 is removed by it.")
    print(f"[setup] {len(SEEDS)} seeds x {len(ALPHAS)} alphas x {len(FRACTIONS)} "
          f"fractions x {N_PROBES} probes\n")

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=cfg["training"]["seq_len"])
    all_windows = [np.asarray(w, dtype=np.float32) for w in ds.sequences]
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(all_windows) * cfg["training"]["val_fraction"]))
    perm = torch.randperm(len(all_windows), generator=g).tolist()
    train_windows = [all_windows[i] for i in perm[:len(all_windows) - vs]]
    eval_windows = [all_windows[i] for i in perm[len(all_windows) - vs:][:2000]]

    target = PAMPOSTarget(
        checkpoint_path=REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
        feature_stats_path=REPO_ROOT / "data" / "processed" / "feature_stats.npz",
        model_config={
            "input_dim": cfg["model"]["input_dim"],
            "d_model": cfg["model"]["d_model"],
            "nhead": cfg["model"]["nhead"],
            "num_layers": cfg["model"]["num_layers"],
            "dim_feedforward": cfg["model"]["dim_feedforward"],
            "dropout": cfg["model"]["dropout"],
        },
        device=device)

    trials = []
    for seed in SEEDS:
        print(f"[seed {seed}] running...")
        r = run_seed(target, train_windows, eval_windows, seed)
        if r:
            trials.append(r)

    if not trials:
        print("[abort] no usable trials")
        return

    print()
    print("=" * 94)
    print("ADAPTIVE ADVERSARY -- attack success (%), mean over seeds")
    print("=" * 94)
    print("Rows: injection strength as a multiple of the filter's z-threshold.")
    print("An alpha below 1 is, by construction, invisible to the filter.\n")
    print(f"{'alpha':<10}", end="")
    for frac in FRACTIONS:
        print(f"{f'{int(frac*100)}% poisoned':<20}", end="")
    print()
    print("-" * 94)

    table = {}
    for alpha in ALPHAS:
        print(f"{alpha:<10}", end="")
        for frac in FRACTIONS:
            key = f"{alpha}_{frac}"
            succ = np.mean([t["cells"][key]["attack_success"] for t in trials])
            rem = np.mean([t["cells"][key]["filter_removal"] for t in trials])
            fa = np.mean([t["cells"][key]["false_alarm"] for t in trials])
            table[key] = {"success": succ, "removal": rem, "false_alarm": fa}
            print(f"{f'{succ:5.1f}  (rm {rem:4.1f}%)':<20}", end="")
        print()
    print("-" * 94)
    print("  rm = share of injected points the filter removed")

    print()
    print("=" * 94)
    print("FALSE-ALARM COST OF EACH CELL (%)")
    print("=" * 94)
    print(f"{'alpha':<10}", end="")
    for frac in FRACTIONS:
        print(f"{f'{int(frac*100)}% poisoned':<20}", end="")
    print()
    print("-" * 94)
    for alpha in ALPHAS:
        print(f"{alpha:<10}", end="")
        for frac in FRACTIONS:
            print(f"{table[f'{alpha}_{frac}']['false_alarm']:<20.2f}", end="")
        print()
    print("-" * 94)

    best = max(table.items(), key=lambda kv: kv[1]["success"])
    evasive = {k: v for k, v in table.items() if v["removal"] < 5.0}
    best_evasive = max(evasive.items(), key=lambda kv: kv[1]["success"]) if evasive else None

    print()
    print("=" * 94)
    print("READING")
    print("=" * 94)
    print(f"  best cell overall          : alpha {best[1] if False else best[0].split('_')[0]}, "
          f"{float(best[0].split('_')[1])*100:.0f}% poisoned -> "
          f"{best[1]['success']:.1f}% success, {best[1]['removal']:.1f}% removed")
    if best_evasive:
        a, f_ = best_evasive[0].split("_")
        print(f"  best cell the filter misses: alpha {a}, {float(f_)*100:.0f}% poisoned -> "
              f"{best_evasive[1]['success']:.1f}% success at "
              f"{best_evasive[1]['false_alarm']:.2f}% false alarms")
        print()
        if best_evasive[1]["success"] > 50:
            print("  An attacker who scales below the filter's threshold defeats the")
            print("  defence. This is a real limitation and must be reported as one,")
            print("  together with the false-alarm cost it imposes on the attacker.")
        elif best_evasive[1]["success"] > 20:
            print("  Sub-threshold injection recovers part of the attack. The defence")
            print("  degrades against an adaptive adversary rather than failing outright.")
        else:
            print("  Sub-threshold injection does not recover the attack: perturbations")
            print("  small enough to evade the filter are too small to move the threshold.")
            print("  The defence is robust to this evasion strategy.")
    else:
        print("  No cell evaded the filter.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "adaptive_attacker.json", "w") as f:
        json.dump({"z_threshold": Z_THRESHOLD, "calibration_size": CALIBRATION_SIZE,
                   "n_probes": N_PROBES, "seeds": SEEDS,
                   "alphas": ALPHAS, "fractions": FRACTIONS,
                   "table": table}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/adaptive_attacker.json")


if __name__ == "__main__":
    main()