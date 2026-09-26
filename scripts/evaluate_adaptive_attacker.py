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

Z_LOCAL = 1.5
CALIBRATION_SIZE = 300
FRACTION_POISONED = 0.15
N_PROBES = 20
SEEDS = [0, 1, 2]
ALPHAS = [0.5, 0.7, 0.9, 1.1, 1.5]


def clean_local(windows, z=Z_LOCAL, injected_mask=None):
    W = np.stack(windows)
    n, T, _ = W.shape
    out = W.copy()
    removed = np.zeros((n, T), dtype=bool)
    for t in range(T):
        others = np.delete(W, t, axis=1)
        med = np.median(others, axis=1)
        mad = np.median(np.abs(others - med[:, None, :]), axis=1) + 1e-8
        hit = (np.abs(W[:, t, :] - med) / (1.4826 * mad)).max(axis=1) > z
        out[hit, t, :] = med[hit]
        removed[hit, t] = True
    inj = float(removed[injected_mask].mean()) if (
        injected_mask is not None and injected_mask.any()) else float("nan")
    return [out[i] for i in range(n)], float(removed.mean()), inj


def est_percentile(scores, pct=99.0):
    return float(np.percentile(scores, pct))


def est_median_mad(scores, k=3.0):
    s = np.asarray(scores)
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return float(med + k * 1.4826 * mad)


def fit_k_for_parity(scores, target_threshold):
    s = np.asarray(scores)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-12
    return float((target_threshold - med) / (1.4826 * mad))


def poison(windows, trigger, alpha, fraction, rng):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    n_points = len(windows) * T
    k = max(1, int(fraction * n_points))
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
        w[t] = med + direction * min(1.0, (alpha * Z_LOCAL) / peak)
    return out, mask


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def run_seed(target, train_windows, all_windows, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows),
                     size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
    calib = [train_windows[i] for i in idx]

    clean_scores = score_all(target, calib)
    clean_threshold = est_percentile(clean_scores)
    k_ref = fit_k_for_parity(clean_scores, clean_threshold)

    scores = [target.raw_score(w) for w in all_windows]
    flagged = sorted([(w, s) for w, s in zip(all_windows, scores) if s > clean_threshold],
                     key=lambda p: p[1], reverse=True)
    if not flagged:
        return None
    probes = [w for w, _ in flagged[:N_PROBES]]

    _, baseline_removal, _ = clean_local(calib)

    cells = {}
    for alpha in ALPHAS + [None]:
        key = "full" if alpha is None else str(alpha)
        hidden_undef = hidden_def = 0
        removals = []
        for probe in probes:
            trigger = probe.mean(axis=0)
            poisoned, mask = poison(calib, trigger, alpha, FRACTION_POISONED, rng)

            sc = score_all(target, poisoned)
            if target.raw_score(probe) <= est_percentile(sc):
                hidden_undef += 1

            cleaned, _, inj_removed = clean_local(poisoned, injected_mask=mask)
            removals.append(inj_removed)
            sc_c = score_all(target, cleaned)
            if target.raw_score(probe) <= est_median_mad(sc_c, k=k_ref):
                hidden_def += 1

        cells[key] = {
            "alpha": alpha,
            "undefended": 100.0 * hidden_undef / len(probes),
            "defended": 100.0 * hidden_def / len(probes),
            "injected_removed": 100.0 * float(np.nanmean(removals)),
        }
        print(f"[seed {seed}] {key} done", flush=True)

    return {"seed": seed, "k_ref": k_ref,
            "baseline_removal": 100.0 * baseline_removal, "cells": cells}


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] device {device}")
    print(f"[setup] protocol matched to evaluate_calibration_methods.py:")
    print(f"[setup]   trigger = probe.mean(axis=0), fraction {FRACTION_POISONED},")
    print(f"[setup]   filter z = {Z_LOCAL}, defended estimator = median+MAD at k_ref")
    print(f"[setup] the only thing varied is how far each injected point is moved")
    print(f"[setup] toward the trigger, expressed as a z-score.\n")

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=cfg["training"]["seq_len"])
    W = [np.asarray(w, dtype=np.float32) for w in ds.sequences]
    g = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(W) * cfg["training"]["val_fraction"]))
    perm = torch.randperm(len(W), generator=g).tolist()
    train_windows = [W[i] for i in perm[:len(W) - vs]]
    eval_windows = [W[i] for i in perm[len(W) - vs:][:2000]]

    target = PAMPOSTarget(
        checkpoint_path=REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
        feature_stats_path=REPO_ROOT / "data" / "processed" / "feature_stats.npz",
        model_config={"input_dim": cfg["model"]["input_dim"],
                      "d_model": cfg["model"]["d_model"],
                      "nhead": cfg["model"]["nhead"],
                      "num_layers": cfg["model"]["num_layers"],
                      "dim_feedforward": cfg["model"]["dim_feedforward"],
                      "dropout": cfg["model"]["dropout"]},
        device=device)

    trials = [r for r in (run_seed(target, train_windows, eval_windows, s)
                          for s in SEEDS) if r]
    if not trials:
        print("[abort] no usable trials")
        return

    base_rm = float(np.mean([t["baseline_removal"] for t in trials]))

    print()
    print("=" * 92)
    print("ADAPTIVE ADVERSARY AGAINST THE TEMPORAL-CONSISTENCY FILTER")
    print("=" * 92)
    print(f"Every row uses the same {int(FRACTION_POISONED * 100)}% poisoning budget, the same probes,")
    print(f"and the same trigger derivation. 'full' is the attack as evaluated in")
    print(f"the paper. A lower alpha moves each injected point less far, staying")
    print(f"under the filter's z-threshold of {Z_LOCAL} and surviving it.")
    print()
    print(f"{'injection':<16}{'undefended':<16}{'defended':<18}{'injected removed'}")
    print("-" * 92)

    table = {}
    for key in [str(a) for a in ALPHAS] + ["full"]:
        u = float(np.mean([t["cells"][key]["undefended"] for t in trials]))
        d = float(np.mean([t["cells"][key]["defended"] for t in trials]))
        sd = float(np.std([t["cells"][key]["defended"] for t in trials]))
        r = float(np.mean([t["cells"][key]["injected_removed"] for t in trials]))
        table[key] = {"undefended": u, "defended": d, "defended_sd": sd,
                      "injected_removed": r}
        label = "full (paper)" if key == "full" else key
        print(f"{label:<16}{u:<16.1f}{f'{d:.1f} +/- {sd:.1f}':<18}{r:.1f}")
    print("-" * 92)
    print(f"  The filter also replaces {base_rm:.1f}% of timesteps in clean calibration")
    print(f"  data, so a removal rate near that figure is not selective.")

    print()
    print("=" * 92)
    print("READING")
    print("=" * 92)
    full = table["full"]
    sub = {a: v for a, v in table.items() if a != "full" and float(a) < 1.0}
    best = max(sub.items(), key=lambda kv: kv[1]["defended"])
    print(f"  paper's attack, defended  : {full['defended']:.1f}% "
          f"({full['injected_removed']:.1f}% of the injection removed)")
    print(f"  best sub-threshold attack : alpha {best[0]}, {best[1]['defended']:.1f}% "
          f"({best[1]['injected_removed']:.1f}% removed)")
    print()
    margin = best[1]["defended"] - full["defended"]
    if margin > 15:
        print(f"  Scaling below the filter threshold recovers {margin:.0f} percentage points")
        print(f"  of attack success. The defence degrades against an adaptive adversary")
        print(f"  and this must be reported as a limitation.")
    elif margin > 5:
        print(f"  Sub-threshold injection gives the attacker {margin:.0f} percentage points")
        print(f"  over the non-adaptive attack. Report the margin.")
    else:
        print(f"  Sub-threshold injection gives no advantage ({margin:+.0f} points): an")
        print(f"  injection small enough to evade the filter is too small to move the")
        print(f"  threshold. The defence resists this evasion strategy.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "adaptive_attacker.json", "w") as f:
        json.dump({"z_local": Z_LOCAL, "fraction": FRACTION_POISONED,
                   "calibration_size": CALIBRATION_SIZE, "n_probes": N_PROBES,
                   "seeds": SEEDS, "baseline_removal": base_rm,
                   "table": table}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/adaptive_attacker.json")


if __name__ == "__main__":
    main()