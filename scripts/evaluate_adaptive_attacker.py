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
FRACTIONS = [0.15]
ALPHA_FULL = None


def filter_windows(windows, injected_mask=None, z_threshold=Z_THRESHOLD):
    """Returns the cleaned windows, the overall removal rate, and -- when a
    mask of injected locations is supplied -- the share of INJECTED points
    the filter removed. The two are very different: the filter also removes
    naturally extreme timesteps from clean data."""
    W = np.stack(windows)
    n, T, F = W.shape
    cleaned = W.copy()
    removed_any = np.zeros((n, T), dtype=bool)
    for t in range(T):
        others = np.delete(W, t, axis=1)
        med = np.median(others, axis=1)
        mad = np.median(np.abs(others - med[:, None, :]), axis=1) + 1e-8
        z = np.abs(W[:, t, :] - med) / (1.4826 * mad)
        hit = z.max(axis=1) > z_threshold
        cleaned[hit, t, :] = med[hit]
        removed_any[hit, t] = True

    overall = float(removed_any.mean())
    injected_removed = float("nan")
    if injected_mask is not None and injected_mask.any():
        injected_removed = float(removed_any[injected_mask].mean())
    return [cleaned[i] for i in range(n)], overall, injected_removed


def z_of_point(window, t, candidate):
    others = np.delete(window, t, axis=0)
    med = np.median(others, axis=0)
    mad = np.median(np.abs(others - med), axis=0) + 1e-8
    return float(np.max(np.abs(candidate - med) / (1.4826 * mad)))


def poison(windows, trigger, alpha, fraction, rng):
    """alpha is the injection strength as a multiple of the filter threshold.
    alpha=None reproduces the non-adaptive attack: the point is replaced by
    the trigger outright, with no regard for the filter."""
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    n_points = len(windows) * T
    k = int(n_points * fraction)
    locs = rng.choice(n_points, size=k, replace=False)
    mask = np.zeros((len(windows), T), dtype=bool)

    for loc in locs:
        wi, t = divmod(loc, T)
        w = out[wi]
        mask[wi, t] = True
        if alpha is None:
            w[t] = trigger
            continue
        others = np.delete(w, t, axis=0)
        med = np.median(others, axis=0)
        mad = np.median(np.abs(others - med), axis=0) + 1e-8
        direction = trigger - med
        peak = float((np.abs(direction) / (1.4826 * mad)).max())
        if peak < 1e-9:
            continue
        scale = min(1.0, (alpha * Z_THRESHOLD) / peak)
        w[t] = med + direction * scale
    return out, mask


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
    probes = [(w, s) for w, s in flagged[:N_PROBES]]
    benign_scores = [target.raw_score(w) for w in calib]

    out = {"seed": seed, "clean_threshold": clean_threshold, "cells": {}}

    _, baseline_removal, _ = filter_windows(calib)
    out["baseline_removal"] = 100.0 * baseline_removal

    for alpha in ALPHAS + [ALPHA_FULL]:
        for frac in FRACTIONS:
            trigger = build_trigger_from_distribution(calib, rng)
            poisoned, mask = poison(calib, trigger, alpha, frac, rng)

            for defended in (False, True):
                if defended:
                    used, _, inj_removed = filter_windows(poisoned, injected_mask=mask)
                else:
                    used, inj_removed = poisoned, 0.0

                target.feature_mae = None
                target.calibrate(used)
                thr = target.threshold
                hidden = sum(1 for _, s in probes if s <= thr)
                fa = float(np.mean([s > thr for s in benign_scores]))

                key = f"{'full' if alpha is None else alpha}_{frac}_{'def' if defended else 'undef'}"
                out["cells"][key] = {
                    "alpha": alpha, "fraction": frac, "defended": defended,
                    "attack_success": 100.0 * hidden / len(probes),
                    "injected_removed": 100.0 * inj_removed,
                    "false_alarm": 100.0 * fa,
                    "threshold": thr,
                }
        print(f"[seed {seed}] alpha {'full' if alpha is None else alpha} done", flush=True)

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

    base_rm = np.mean([t["baseline_removal"] for t in trials])
    print()
    print("=" * 94)
    print("ADAPTIVE ADVERSARY -- all cells at the same 15% poisoning budget")
    print("=" * 94)
    print("alpha is the injection strength as a multiple of the filter's")
    print(f"z-threshold ({Z_THRESHOLD}). Below 1 the injected point cannot be removed.")
    print("'full' is the non-adaptive attack: the point is replaced by the trigger")
    print("outright, ignoring the filter.")
    print()
    print(f"For reference, the filter replaces {base_rm:.1f}% of timesteps in CLEAN")
    print("calibration data, so a removal rate near that figure means the filter")
    print("is not singling the injection out.")
    print()
    print(f"{'injection':<14}{'undefended':<16}{'defended':<16}"
          f"{'injected removed':<20}{'false alarms'}")
    print("-" * 94)

    table = {}
    order = [str(a) for a in ALPHAS] + ["full"]
    for a in order:
        frac = FRACTIONS[0]
        ku, kd = f"{a}_{frac}_undef", f"{a}_{frac}_def"
        if ku not in trials[0]["cells"]:
            continue
        undef = np.mean([t["cells"][ku]["attack_success"] for t in trials])
        dfn = np.mean([t["cells"][kd]["attack_success"] for t in trials])
        rm = np.mean([t["cells"][kd]["injected_removed"] for t in trials])
        fa = np.mean([t["cells"][kd]["false_alarm"] for t in trials])
        table[a] = {"undefended": undef, "defended": dfn,
                    "injected_removed": rm, "false_alarm": fa}
        label = a if a != "full" else "full (paper)"
        print(f"{label:<14}{undef:<16.1f}{dfn:<16.1f}{rm:<20.1f}{fa:.2f}")
    print("-" * 94)

    print()
    print("=" * 94)
    print("READING")
    print("=" * 94)
    sub = {a: v for a, v in table.items() if a != "full" and float(a) < 1.0}
    if sub and "full" in table:
        best_sub = max(sub.items(), key=lambda kv: kv[1]["defended"])
        full = table["full"]
        print(f"  non-adaptive attack, defended : {full['defended']:.1f}% "
              f"({full['injected_removed']:.1f}% of its injection removed)")
        print(f"  best sub-threshold attack     : alpha {best_sub[0]}, "
              f"{best_sub[1]['defended']:.1f}% "
              f"({best_sub[1]['injected_removed']:.1f}% removed)")
        print()
        if best_sub[1]["defended"] > full["defended"] + 15:
            print("  Scaling below the filter threshold recovers a substantial part of")
            print("  the attack. The defence degrades against an adaptive adversary and")
            print("  this must be reported as a limitation.")
        elif best_sub[1]["defended"] > full["defended"] + 5:
            print("  Sub-threshold injection gives the attacker a modest advantage over")
            print("  the non-adaptive attack. Report the margin.")
        else:
            print("  Sub-threshold injection gives the attacker no advantage: a")
            print("  perturbation small enough to evade the filter is too small to move")
            print("  the threshold. The defence is robust to this evasion strategy.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "adaptive_attacker.json", "w") as f:
        json.dump({"z_threshold": Z_THRESHOLD, "calibration_size": CALIBRATION_SIZE,
                   "n_probes": N_PROBES, "seeds": SEEDS,
                   "alphas": ALPHAS, "fraction": FRACTIONS[0],
                   "baseline_removal": float(base_rm),
                   "table": table}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/adaptive_attacker.json")


if __name__ == "__main__":
    main()