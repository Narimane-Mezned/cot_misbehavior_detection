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
CALIBRATION_SIZE = 300
FRACTIONS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
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


def run(target, train_windows, all_windows, fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows),
                     size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
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
        poisoned = poison(calib, probe.mean(axis=0), fraction, rng)

        sc = score_all(target, poisoned)
        if target.raw_score(probe) <= est_percentile(sc):
            undef += 1

        sc_c = score_all(target, clean_local(poisoned))
        if target.raw_score(probe) <= est_median_mad(sc_c, k_ref):
            dfn += 1

    fa_clean = 100.0 * float(np.mean([s > clean_thr for s in scores]))

    n_injected = max(1, int(fraction * len(calib) * calib[0].shape[0]))
    return {"undefended": 100.0 * undef / len(probes),
            "n_injected": n_injected,
            "defended": 100.0 * dfn / len(probes),
            "threshold": clean_thr, "k": k_ref,
            "false_alarm_clean": fa_clean, "n_flagged": len(flagged)}


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] device {device}")
    print(f"[setup] sweeping the poisoning rate over "
          f"{[f'{int(f*100)}%' for f in FRACTIONS]}")
    print(f"[setup] {len(SEEDS)} seeds x {N_PROBES} probes at each rate")
    print(f"[setup] everything else held fixed: {CALIBRATION_SIZE}-window calibration,")
    print(f"[setup] filter z={Z_LOCAL}, median+MAD at k fitted for parity")
    print(f"[setup] the paper reports 15%; three reviews asked what happens either side\n")

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
    for frac in FRACTIONS:
        trials = [r for r in (run(target, train_windows, all_windows, frac, s)
                              for s in SEEDS) if r]
        if not trials:
            continue
        results[frac] = {
            key: (float(np.mean([t[key] for t in trials])),
                  float(np.std([t[key] for t in trials])))
            for key in ["undefended", "defended", "threshold",
                        "false_alarm_clean", "n_flagged", "n_injected"]
        }
        u = results[frac]["undefended"]
        d = results[frac]["defended"]
        print(f"[rate {int(frac*100):>3}%] {int(results[frac]['n_injected'][0]):>4} points  "
              f"undefended {u[0]:5.1f} +/- {u[1]:4.1f}   "
              f"defended {d[0]:5.1f} +/- {d[1]:4.1f}", flush=True)

    print()
    print("=" * 88)
    print("ATTACK AND DEFENCE VERSUS CONTAMINATION RATE")
    print("=" * 88)
    print(f"{'poisoned':<12}{'points':<9}{'undefended':<18}{'defended':<18}"
          f"{'benign FA %'}")
    print("-" * 88)
    for frac in FRACTIONS:
        if frac not in results:
            continue
        r = results[frac]
        u_txt = f"{r['undefended'][0]:.1f} +/- {r['undefended'][1]:.1f}"
        d_txt = f"{r['defended'][0]:.1f} +/- {r['defended'][1]:.1f}"
        mark = "  <- paper" if abs(frac - 0.15) < 1e-9 else ""
        print(f"{f'{int(frac*100)}%':<12}{int(r['n_injected'][0]):<9}{u_txt:<18}"
              f"{d_txt:<18}{r['false_alarm_clean'][0]:.2f}{mark}")
    print("-" * 88)

    print()
    print("=" * 88)
    print("READING")
    print("=" * 88)
    und = [results[f]["undefended"][0] for f in FRACTIONS if f in results]
    dfn = [results[f]["defended"][0] for f in FRACTIONS if f in results]
    lo = min(f for f in FRACTIONS if f in results)
    print(f"  undefended success spans {min(und):.1f} to {max(und):.1f}%")
    print(f"  defended   success spans {min(dfn):.1f} to {max(dfn):.1f}%")
    print()
    if results[lo]["undefended"][0] > 80:
        print(f"  The attack already succeeds at {results[lo]['undefended'][0]:.0f}% with only")
        print(f"  {int(results[lo]['n_injected'][0])} injected points ({int(lo*100)}% of the")
        print(f"  calibration set). The vulnerability does not require a large")
        print(f"  contamination budget, which strengthens the supply-chain threat model.")
    else:
        print(f"  At {int(lo*100)}% contamination the attack reaches only")
        print(f"  {results[lo]['undefended'][0]:.1f}%, so it does require a substantial")
        print(f"  budget. Report the threshold at which it becomes effective.")
    print()
    if max(dfn) < 30:
        print(f"  The defence holds across the whole range tested, never exceeding")
        print(f"  {max(dfn):.1f}% attacker success.")
    else:
        print(f"  The defence degrades as contamination rises, reaching {max(dfn):.1f}%")
        print(f"  at the highest rate tested. This is a limitation and must be reported.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "contamination_rate_sweep.json", "w") as f:
        json.dump({"fractions": FRACTIONS, "seeds": SEEDS, "n_probes": N_PROBES,
                   "calibration_size": CALIBRATION_SIZE, "z_local": Z_LOCAL,
                   "results": {str(k): v for k, v in results.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/calibration_size_sweep.json")


if __name__ == "__main__":
    main()