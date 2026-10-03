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
RATES = [0.01, 0.05, 0.15, 0.30]
PGD_STEPS = 40
PGD_LR = 0.15
N_CLEAN_TRIALS = 30
N_SEEDS = 3
K_PARITY_PATH = "outputs/results/calibration_method_comparison.json"


def est_percentile(s, pct=99.0):
    return float(np.percentile(s, pct))


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def _box(window, t, z=Z_LOCAL):
    others = np.delete(window, t, axis=0)
    med = np.median(others, axis=0)
    mad = np.median(np.abs(others - med), axis=0) + 1e-8
    half = z * 1.4826 * mad
    return med - half, med + half


def poison(windows, trigger, fraction, rng, mode="full",
           target=None, mean=None, std=None, device=None):
    """mode "full" repeats one trigger vector at every injected point.
    "sub-threshold" and "white-box" are the adaptive attackers of
    Section 5.3: each injected point is distinct and stays inside the
    filter's feasible box. They matter here because a repeated constant
    distorts the per-feature error scale far more than a set of distinct
    values does, and a drift monitor that caught only the constant would be
    attack-specific."""
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    k = max(1, int(fraction * len(windows) * T))
    locs = rng.choice(len(windows) * T, size=k, replace=False)

    if mode == "full":
        for loc in locs:
            wi, t = divmod(loc, T)
            out[wi][t] = trigger
        return out

    if mode == "sub-threshold":
        for loc in locs:
            wi, t = divmod(loc, T)
            w = out[wi]
            lo, hi = _box(w, t)
            med = 0.5 * (lo + hi)
            w[t] = np.clip(med + (trigger - med) * 0.9, lo, hi)
        return out

    from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
    by_window = {}
    for loc in locs:
        wi, t = divmod(loc, T)
        by_window.setdefault(wi, []).append(t)
    for wi, ts in by_window.items():
        w = out[wi]
        boxes = {t: _box(w, t) for t in ts}
        x = torch.tensor((w - mean) / std, dtype=torch.float32,
                         device=device, requires_grad=True)
        idx = torch.tensor(ts, dtype=torch.long, device=device)
        lo = torch.stack([torch.tensor((boxes[t][0] - mean) / std,
                                       dtype=torch.float32, device=device) for t in ts])
        hi = torch.stack([torch.tensor((boxes[t][1] - mean) / std,
                                       dtype=torch.float32, device=device) for t in ts])
        for _ in range(PGD_STEPS):
            if x.grad is not None:
                x.grad.zero_()
            preds = target.model(x.unsqueeze(0)[:, :-1, :])
            err = per_feature_errors(preds, x.unsqueeze(0)[:, 1:, :])
            if target.feature_mae is not None:
                err = normalize_errors(err, target.feature_mae)
            topk_anomaly_score(err, k=3).mean().backward()
            with torch.no_grad():
                upd = x.clone()
                upd[idx] = torch.max(torch.min(x[idx] + PGD_LR * torch.sign(x.grad[idx]),
                                               hi), lo)
                x = upd.detach().requires_grad_(True)
        out[wi] = (x.detach().cpu().numpy() * std + mean).astype(np.float32)
    return out


def drift_statistics(threshold, mae, base_threshold, base_mae):
    """Several candidate statistics, all computable by a deployment from its
    own stored calibration record and nothing else.

    max-relative was the first attempt: the larger of the threshold and scale
    displacements. It is dominated by the scale term, which a repeated
    constant inflates enormously but which distinct injected values barely
    move, so it misses the adaptive attackers at low contamination.

    The others are directional. Poisoning can only raise the threshold, since
    it adds badly predicted points to the upper tail; ordinary resampling
    moves it either way. A signed test therefore discards half the clean
    variation that a two-sided one has to tolerate."""
    t, bt = float(threshold), float(base_threshold)
    m, bm = np.asarray(mae, dtype=float), np.asarray(base_mae, dtype=float)

    d_thr_abs = abs(t - bt) / max(bt, 1e-9)
    d_thr_signed = (t - bt) / max(bt, 1e-9)
    d_scale = float(np.max(np.abs(m - bm) / np.maximum(bm, 1e-9)))
    d_scale_signed = float(np.max((m - bm) / np.maximum(bm, 1e-9)))

    return {
        "max-relative": max(d_thr_abs, d_scale),
        "threshold-signed": d_thr_signed,
        "scale-signed": d_scale_signed,
        "sum-signed": d_thr_signed + d_scale_signed,
    }


STATISTICS = ["max-relative", "threshold-signed", "scale-signed", "sum-signed"]
ALARM_PCTS = [90.0, 95.0]


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
        clean.append(drift_statistics(thr, target.feature_mae.cpu().numpy(),
                                      base_thr, base_mae))

    clean_by_stat = {st: np.asarray([c[st] for c in clean]) for st in STATISTICS}
    alarms = {}
    print(f"[clean] over {N_CLEAN_TRIALS} ordinary recalibrations:")
    for st in STATISTICS:
        v = clean_by_stat[st]
        for pct in ALARM_PCTS:
            a = float(np.percentile(v, pct))
            alarms[(st, pct)] = a
            fpr = 100.0 * float((v > a).mean())
            print(f"   {st:<18} p{int(pct)} alarm {a:9.4f}  "
                  f"(clean median {np.median(v):8.4f}, "
                  f"false positives {fpr:4.1f}%)")
    print(flush=True)

    # ---- poisoned recalibrations ---------------------------------------
    scores = np.asarray([target.raw_score(w) for w in all_windows])
    order = np.argsort(-scores)
    probe = np.asarray(all_windows[order[0]], dtype=np.float32)
    trigger = probe.mean(axis=0)

    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)

    results = {}
    for mode in ["full", "sub-threshold", "white-box"]:
        for rate in RATES:
            per_seed = []
            for si in range(N_SEEDS):
                rng = np.random.default_rng(2000 + si)
                idx = rng.choice(len(train_windows), size=CALIBRATION_SIZE,
                                 replace=False)
                cal = [np.asarray(train_windows[j], dtype=np.float32) for j in idx]
                pois = poison(cal, trigger, rate, rng, mode,
                              target, mean, std, device)
                thr = est_percentile(score_all(target, pois))
                per_seed.append(drift_statistics(
                    thr, target.feature_mae.cpu().numpy(), base_thr, base_mae))
            for st in STATISTICS:
                v = np.asarray([p[st] for p in per_seed])
                for pct in ALARM_PCTS:
                    results[(mode, rate, st, pct)] = {
                        "drift": float(np.mean(v)), "sd": float(np.std(v)),
                        "detected": 100.0 * float((v > alarms[(st, pct)]).mean())}
            best = max(STATISTICS,
                       key=lambda st: results[(mode, rate, st, 90.0)]["detected"])
            print(f"[{mode:<14}{int(rate*100):>3}%] "
                  f"best statistic {best:<18} "
                  f"flagged {results[(mode, rate, best, 90.0)]['detected']:3.0f}% "
                  f"at p90", flush=True)
        print(flush=True)

    print()
    print("=" * 80)
    print("DETECTING A POISONED RECALIBRATION FROM DRIFT ALONE")
    print("=" * 80)
    print(f"Alarm level {alarm:.4f}, set at the 95th percentile of "
          f"{N_CLEAN_TRIALS} ordinary")
    print(f"recalibrations. No reference score and no external information "
          f"are used.")
    print()
    for pct in ALARM_PCTS:
        print()
        print(f"  alarm at the {int(pct)}th percentile of clean recalibration")
        print("  " + "-" * 76)
        hdr = "  %-16s%-8s" % ("attacker", "rate")
        for st in STATISTICS:
            hdr += "%-15s" % st.replace("-", "-\n")[:14]
        print("  %-16s%-8s%s" % ("attacker", "rate",
                                 "".join("%-15s" % st[:14] for st in STATISTICS)))
        print("  " + "-" * 76)
        for mode in ["full", "sub-threshold", "white-box"]:
            for rate in RATES:
                row = "  %-16s%-8s" % (mode if rate == RATES[0] else "",
                                       "%d%%" % int(rate * 100))
                for st in STATISTICS:
                    row += "%-15s" % ("%.0f%%" % results[(mode, rate, st, pct)]["detected"])
                print(row)
        print("  " + "-" * 76)
        fprs = []
        for st in STATISTICS:
            v = clean_by_stat[st]
            fprs.append("%.0f%%" % (100.0 * float((v > alarms[(st, pct)]).mean())))
        print("  %-16s%-8s%s" % ("false positives", "", "".join("%-15s" % x for x in fprs)))

    print()
    print("=" * 80)
    print("READING")
    print("=" * 80)
    best = None
    for st in STATISTICS:
        for pct in ALARM_PCTS:
            worst = min(results[(m, r, st, pct)]["detected"]
                        for m in ["full", "sub-threshold", "white-box"]
                        for r in RATES)
            fpr = 100.0 * float((clean_by_stat[st] > alarms[(st, pct)]).mean())
            if best is None or (worst, -fpr) > (best[2], -best[3]):
                best = (st, pct, worst, fpr)
    st, pct, worst, fpr = best
    print(f"  best configuration: {st} at the {int(pct)}th percentile")
    print(f"  weakest cell {worst:.0f}% detected, "
          f"false positives {fpr:.0f}% on clean recalibration")
    print()
    if worst >= 100:
        print("  Every poisoned recalibration is flagged, for every attacker and")
        print("  every rate tested, including 1% contamination. The explanation")
        print("  layer can raise the integrity warning from the detector's own")
        print("  stored record, with no external reference.")
    elif worst > 0:
        print("  Detection is incomplete at the lowest contamination. Report the")
        print("  bound: the monitor is reliable above that rate and blind below it.")
    else:
        print("  No statistic tested separates the adaptive attackers at the")
        print("  lowest rate. A drift monitor is not sufficient on its own, and")
        print("  the limitation stands with evidence that four statistics and")
        print("  two alarm levels were tried.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "drift_detection.json", "w") as f:
        json.dump({"baseline_threshold": base_thr, "alarms": {f"{st}_{pct}": v for (st, pct), v in alarms.items()},
                   "clean_medians": {st: float(np.median(v)) for st, v in clean_by_stat.items()},
                   "n_clean_trials": N_CLEAN_TRIALS, "n_seeds": N_SEEDS,
                   "results": {f"{m}_{r}_{st}_{pct}": v for (m, r, st, pct), v in results.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/drift_detection.json")


if __name__ == "__main__":
    main()