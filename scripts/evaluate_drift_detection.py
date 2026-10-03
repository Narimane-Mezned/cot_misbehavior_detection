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
RATES = [0.01, 0.05, 0.10, 0.15, 0.30]
N_CLEAN_TRIALS = 30
N_SEEDS = 3
K_PARITY_PATH = "outputs/results/calibration_method_comparison.json"


def est_percentile(s, pct=99.0):
    return float(np.percentile(s, pct))


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def poison(windows, trigger, fraction, rng):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    k = max(1, int(fraction * len(windows) * T))
    for loc in rng.choice(len(windows) * T, size=k, replace=False):
        wi, t = divmod(loc, T)
        out[wi][t] = trigger
    return out


def drift_statistic(threshold, mae, base_threshold, base_mae):
    """How far a fresh calibration has moved from the stored baseline.

    The threshold alone is not enough: a poisoned calibration inflates it, but
    so does an unusually eventful sample of ordinary traffic. The per-feature
    error scale moves differently in the two cases, because poisoning inflates
    the tail without changing the bulk. We therefore take the larger of the
    two relative displacements, which is what a deployment could compute from
    its own stored calibration record and nothing else."""
    d_thr = abs(threshold - base_threshold) / max(base_threshold, 1e-9)
    d_mae = float(np.max(np.abs(np.asarray(mae) - np.asarray(base_mae))
                         / np.maximum(np.asarray(base_mae), 1e-9)))
    return max(d_thr, d_mae), d_thr, d_mae


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] device {device}")
    print(f"[setup] A deployment stores its calibration record: the threshold")
    print(f"[setup] and the per-feature error scale. When it recalibrates, it")
    print(f"[setup] can compare the new record against the stored one without")
    print(f"[setup] any external reference. This measures whether that drift")
    print(f"[setup] alone separates a poisoned recalibration from an ordinary")
    print(f"[setup] one, which is what the explanation layer would need in")
    print(f"[setup] order to flag compromise for itself.")
    print(f"[setup] {N_CLEAN_TRIALS} clean recalibrations, "
          f"{N_SEEDS} seeds x {len(RATES)} poisoning rates\n")

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=cfg["training"]["seq_len"])
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=gen)
    train_windows = [ds.sequences[i] for i in tr.indices]
    all_windows = [ds.sequences[i] for i in range(len(ds))]

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

    # ---- the stored baseline, from the canonical calibration ----------
    rng0 = np.random.default_rng(0)
    base_idx = rng0.choice(len(train_windows), size=CALIBRATION_SIZE, replace=False)
    base_calib = [np.asarray(train_windows[i], dtype=np.float32) for i in base_idx]
    base_thr = est_percentile(score_all(target, base_calib))
    base_mae = target.feature_mae.cpu().numpy().copy()
    print(f"[baseline] threshold {base_thr:.4f}\n")

    # ---- what ordinary recalibration looks like ------------------------
    clean = []
    for i in range(N_CLEAN_TRIALS):
        rng = np.random.default_rng(1000 + i)
        idx = rng.choice(len(train_windows), size=CALIBRATION_SIZE, replace=False)
        cal = [np.asarray(train_windows[j], dtype=np.float32) for j in idx]
        thr = est_percentile(score_all(target, cal))
        d, dt, dm = drift_statistic(thr, target.feature_mae.cpu().numpy(),
                                    base_thr, base_mae)
        clean.append(d)
    clean = np.asarray(clean)
    print(f"[clean] drift over {N_CLEAN_TRIALS} ordinary recalibrations: "
          f"median {np.median(clean):.4f}, max {clean.max():.4f}")

    # the alarm level: a deployment would set this from its own history
    alarm = float(np.percentile(clean, 95))
    fp = 100.0 * float((clean > alarm).mean())
    print(f"[clean] alarm level at the 95th percentile of clean drift: "
          f"{alarm:.4f}  ({fp:.1f}% of clean recalibrations exceed it)\n")

    # ---- poisoned recalibrations ---------------------------------------
    scores = np.asarray([target.raw_score(w) for w in all_windows])
    order = np.argsort(-scores)
    probe = np.asarray(all_windows[order[0]], dtype=np.float32)
    trigger = probe.mean(axis=0)

    results = {}
    for rate in RATES:
        drifts = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(2000 + s)
            idx = rng.choice(len(train_windows), size=CALIBRATION_SIZE, replace=False)
            cal = [np.asarray(train_windows[j], dtype=np.float32) for j in idx]
            pois = poison(cal, trigger, rate, rng)
            thr = est_percentile(score_all(target, pois))
            d, dt, dm = drift_statistic(thr, target.feature_mae.cpu().numpy(),
                                        base_thr, base_mae)
            drifts.append((d, dt, dm))
        d = np.array([x[0] for x in drifts])
        results[rate] = {"drift": float(np.mean(d)), "sd": float(np.std(d)),
                         "detected": 100.0 * float((d > alarm).mean()),
                         "d_threshold": float(np.mean([x[1] for x in drifts])),
                         "d_scale": float(np.mean([x[2] for x in drifts]))}
        print(f"[rate {int(rate*100):>3}%] drift {np.mean(d):.4f} "
              f"(threshold {results[rate]['d_threshold']:.4f}, "
              f"scale {results[rate]['d_scale']:.4f})  "
              f"flagged {results[rate]['detected']:.0f}%", flush=True)

    print()
    print("=" * 80)
    print("DETECTING A POISONED RECALIBRATION FROM DRIFT ALONE")
    print("=" * 80)
    print(f"Alarm level {alarm:.4f}, set at the 95th percentile of "
          f"{N_CLEAN_TRIALS} ordinary")
    print(f"recalibrations. No reference score and no external information "
          f"are used.")
    print()
    print("%-12s%-16s%-16s%s" % ("poisoned", "drift", "vs alarm", "flagged"))
    print("-" * 80)
    print("%-12s%-16s%-16s%s" % ("none (clean)",
                                 "%.4f" % float(np.median(clean)),
                                 "below", "%.0f%%" % fp))
    for rate in RATES:
        r = results[rate]
        print("%-12s%-16s%-16s%s" % (
            "%d%%" % int(rate * 100),
            "%.4f +/- %.4f" % (r["drift"], r["sd"]),
            "%.1fx" % (r["drift"] / alarm if alarm > 0 else 0),
            "%.0f%%" % r["detected"]))
    print("-" * 80)

    print()
    print("=" * 80)
    print("READING")
    print("=" * 80)
    worst = min(RATES)
    if results[worst]["detected"] >= 100:
        print(f"  Every poisoned recalibration is flagged, down to "
              f"{int(worst*100)}% contamination,")
        print(f"  at a {fp:.0f}% false-positive rate on ordinary recalibration. The")
        print(f"  explanation layer can therefore raise the integrity warning from")
        print(f"  the detector's own stored history, without the reference score")
        print(f"  the experiments above supplied.")
    else:
        lowest = min((r for r in RATES if results[r]["detected"] >= 100), default=None)
        if lowest:
            print(f"  Poisoning is detected reliably from {int(lowest*100)}% upward.")
            print(f"  Below that the drift is within ordinary recalibration variation,")
            print(f"  so the method bounds what it can claim rather than detecting all.")
        else:
            print(f"  Drift does not separate poisoned from ordinary recalibration at")
            print(f"  the rates tested. The limitation stands, now with evidence that")
            print(f"  a threshold-drift monitor was tried and found insufficient.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "drift_detection.json", "w") as f:
        json.dump({"baseline_threshold": base_thr, "alarm": alarm,
                   "clean_median": float(np.median(clean)),
                   "clean_max": float(clean.max()),
                   "false_positive_pct": fp,
                   "n_clean_trials": N_CLEAN_TRIALS, "n_seeds": N_SEEDS,
                   "results": {str(k): v for k, v in results.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/drift_detection.json")


if __name__ == "__main__":
    main()