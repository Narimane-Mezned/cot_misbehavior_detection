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
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import PAMPOSTarget

Z_LOCAL = 1.5
SIZES = [100, 300, 1000, 3000]
FRACTION_POISONED = 0.15
N_PROBES = 20
SEEDS = [0, 1, 2]


def clean_local(windows, z=Z_LOCAL):
    W = np.stack(windows)
    n, T, _ = W.shape
    out = W.copy()
    for t in range(T):
        others = np.delete(W, t, axis=1)
        med = np.median(others, axis=1)
        mad = np.median(np.abs(others - med[:, None, :]), axis=1) + 1e-8
        hit = (np.abs(W[:, t, :] - med) / (1.4826 * mad)).max(axis=1) > z
        out[hit, t, :] = med[hit]
    return [out[i] for i in range(n)]


def est_percentile(scores, pct=99.0):
    return float(np.percentile(scores, pct))


def est_median_mad(scores, k):
    s = np.asarray(scores)
    med = np.median(s)
    return float(med + k * 1.4826 * np.median(np.abs(s - med)))


def fit_k(scores, target_threshold):
    s = np.asarray(scores)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-12
    return float((target_threshold - med) / (1.4826 * mad))


def poison(windows, trigger, fraction, rng):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    n_points = len(windows) * T
    locs = rng.choice(n_points, size=max(1, int(fraction * n_points)), replace=False)
    for loc in locs:
        wi, t = divmod(loc, T)
        out[wi][t] = trigger
    return out


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def run(target, train_windows, all_windows, size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), size=min(size, len(train_windows)), replace=False)
    calib = [np.asarray(train_windows[i], dtype=np.float32) for i in idx]

    clean_scores = score_all(target, calib)
    clean_thr = est_percentile(clean_scores)
    k_ref = fit_k(clean_scores, clean_thr)

    scores = [target.raw_score(w) for w in all_windows]
    flagged = sorted([(w, s) for w, s in zip(all_windows, scores) if s > clean_thr],
                     key=lambda p: p[1], reverse=True)
    if not flagged:
        return None
    probes = [np.asarray(w, dtype=np.float32) for w, _ in flagged[:N_PROBES]]

    undef = dfn = 0
    for probe in probes:
        poisoned = poison(calib, probe.mean(axis=0), FRACTION_POISONED, rng)

        sc = score_all(target, poisoned)
        if target.raw_score(probe) <= est_percentile(sc):
            undef += 1

        sc_c = score_all(target, clean_local(poisoned))
        if target.raw_score(probe) <= est_median_mad(sc_c, k_ref):
            dfn += 1

    fa_clean = 100.0 * float(np.mean([s > clean_thr for s in scores]))

    return {"undefended": 100.0 * undef / len(probes),
            "defended": 100.0 * dfn / len(probes),
            "threshold": clean_thr, "k": k_ref,
            "false_alarm_clean": fa_clean, "n_flagged": len(flagged)}


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] device {device}")
    print(f"[setup] sweeping the calibration set over {SIZES} windows")
    print(f"[setup] {len(SEEDS)} seeds x {N_PROBES} probes at each size")
    print(f"[setup] everything else is held fixed: {int(FRACTION_POISONED*100)}% poisoning,")
    print(f"[setup] filter z={Z_LOCAL}, median+MAD at k fitted for parity\n")

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=cfg["training"]["seq_len"])
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=gen)
    train_windows = [ds.sequences[i] for i in tr.indices]
    all_windows = [ds.sequences[i] for i in range(len(ds))]
    print(f"[setup] {len(train_windows)} train / {len(all_windows)} total\n")

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

    results = {}
    for size in SIZES:
        trials = [r for r in (run(target, train_windows, all_windows, size, s)
                              for s in SEEDS) if r]
        if not trials:
            continue
        results[size] = {
            key: (float(np.mean([t[key] for t in trials])),
                  float(np.std([t[key] for t in trials])))
            for key in ["undefended", "defended", "threshold",
                        "false_alarm_clean", "n_flagged"]
        }
        u = results[size]["undefended"]
        d = results[size]["defended"]
        print(f"[size {size:>5}] undefended {u[0]:5.1f} +/- {u[1]:4.1f}   "
              f"defended {d[0]:5.1f} +/- {d[1]:4.1f}", flush=True)

    print()
    print("=" * 88)
    print("SENSITIVITY TO THE CALIBRATION SET SIZE")
    print("=" * 88)
    print(f"{'windows':<12}{'undefended':<18}{'defended':<18}"
          f"{'threshold':<14}{'benign FA %'}")
    print("-" * 88)
    for size in SIZES:
        if size not in results:
            continue
        r = results[size]
        u_txt = f"{r['undefended'][0]:.1f} +/- {r['undefended'][1]:.1f}"
        d_txt = f"{r['defended'][0]:.1f} +/- {r['defended'][1]:.1f}"
        print(f"{size:<12}{u_txt:<18}{d_txt:<18}"
              f"{r['threshold'][0]:<14.3f}{r['false_alarm_clean'][0]:.2f}")
    print("-" * 88)

    print()
    print("=" * 88)
    print("READING")
    print("=" * 88)
    und = [results[s]["undefended"][0] for s in SIZES if s in results]
    dfn = [results[s]["defended"][0] for s in SIZES if s in results]
    print(f"  undefended attack success ranges {min(und):.1f} to {max(und):.1f}%")
    print(f"  defended   attack success ranges {min(dfn):.1f} to {max(dfn):.1f}%")
    print()
    if min(und) > 80 and max(dfn) < 30:
        print("  Both conclusions hold at every size tested: the attack succeeds")
        print("  against percentile calibration regardless of how much data the")
        print("  threshold is drawn from, and the defence reduces it substantially")
        print("  at every size. The choice of 300 windows is not load-bearing.")
    elif max(dfn) - min(dfn) > 30:
        print("  The defence's effectiveness depends materially on the calibration")
        print("  size. This must be reported, and the size used in the paper")
        print("  justified rather than merely stated.")
    else:
        print("  Report the range as measured.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibration_size_sweep.json", "w") as f:
        json.dump({"sizes": SIZES, "seeds": SEEDS, "n_probes": N_PROBES,
                   "fraction": FRACTION_POISONED, "z_local": Z_LOCAL,
                   "results": {str(k): v for k, v in results.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/calibration_size_sweep.json")


if __name__ == "__main__":
    main()