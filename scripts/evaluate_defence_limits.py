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
PASSES = [1, 2, 3]
RATES_A3 = [0.30, 0.40, 0.50, 0.60, 0.70]
RATE_A6 = 0.15
N_PROBES = 60
SEEDS = [0, 1, 2]
PGD_STEPS = 40
PGD_LR = 0.15
K_PARITY_PATH = "outputs/results/calibration_method_comparison.json"


def filter_passes(W, rounds, z=Z_LOCAL):
    out = W.copy()
    touched = np.zeros(W.shape[:2], dtype=bool)
    for _ in range(rounds):
        n, T, _ = out.shape
        flagged = np.zeros((n, T), dtype=bool)
        med = np.zeros_like(out)
        for t in range(T):
            others = np.delete(out, t, axis=1)
            m = np.median(others, axis=1)
            mad = np.median(np.abs(others - m[:, None, :]), axis=1) + 1e-8
            flagged[:, t] = (np.abs(out[:, t, :] - m) / (1.4826 * mad)).max(axis=1) > z
            med[:, t, :] = m
        out = out.copy()
        out[flagged] = med[flagged]
        touched |= flagged
    return out, float(touched.mean())


def est_percentile(s, pct=99.0):
    return float(np.percentile(s, pct))


def est_median_mad(s, k):
    s = np.asarray(s)
    m = np.median(s)
    return float(m + k * 1.4826 * np.median(np.abs(s - m)))


def box_after_passes(window, t, rounds, z=Z_LOCAL):
    """The feasible region for an injected point at timestep t, computed on the
    window as it will look after `rounds - 1` passes have already repaired it.
    An attacker who knows the filter runs several times would aim at this
    rather than at the box the first pass sees."""
    w = window
    if rounds > 1:
        w, _ = filter_passes(window[None, ...], rounds - 1)
        w = w[0]
    others = np.delete(w, t, axis=0)
    med = np.median(others, axis=0)
    mad = np.median(np.abs(others - med), axis=0) + 1e-8
    half = z * 1.4826 * mad
    return med - half, med + half


def poison_full(windows, trigger, fraction, rng):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    locs = rng.choice(len(windows) * T,
                      size=max(1, int(fraction * len(windows) * T)), replace=False)
    for loc in locs:
        wi, t = divmod(loc, T)
        out[wi][t] = trigger
    return out


def poison_subthreshold(windows, trigger, fraction, rng, rounds, alpha=0.9):
    """Sub-threshold injection aimed at the box the filter will see on its
    final pass, not its first."""
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    locs = rng.choice(len(windows) * T,
                      size=max(1, int(fraction * len(windows) * T)), replace=False)
    for loc in locs:
        wi, t = divmod(loc, T)
        w = out[wi]
        lo, hi = box_after_passes(w, t, rounds)
        med = 0.5 * (lo + hi)
        w[t] = np.clip(med + (trigger - med) * alpha, lo, hi)
    return out


def poison_whitebox(windows, trigger, fraction, rng, rounds,
                    target, mean, std, device):
    """Projected gradient ascent, projected into the box the filter will see
    after the earlier passes have run."""
    from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    locs = rng.choice(len(windows) * T,
                      size=max(1, int(fraction * len(windows) * T)), replace=False)
    by_window = {}
    for loc in locs:
        wi, t = divmod(loc, T)
        by_window.setdefault(wi, []).append(t)

    for wi, ts in by_window.items():
        w = out[wi]
        boxes = {t: box_after_passes(w, t, rounds) for t in ts}
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


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kp = REPO_ROOT / K_PARITY_PATH
    if not kp.exists():
        print(f"[abort] {K_PARITY_PATH} not found")
        return
    k_ref = float(json.load(open(kp))["k_parity"])

    print(f"[setup] device {device}, k = {k_ref:.4f}")
    print(f"[setup] A3: contamination {[f'{int(r*100)}%' for r in RATES_A3]} against")
    print(f"[setup]     1, 2 and 3 passes, to locate the breakdown point.")
    print(f"[setup] A6: the two adaptive adversaries rebuilt against the")
    print(f"[setup]     multi-pass filter. They now aim at the feasible box the")
    print(f"[setup]     filter sees on its LAST pass, not its first, which is")
    print(f"[setup]     what an attacker who knows the pass count would do.")
    print(f"[setup] {len(SEEDS)} seeds x {N_PROBES} probes\n")

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"],
                                   seq_len=cfg["training"]["seq_len"])
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    vs = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=gen)
    train_windows = [ds.sequences[i] for i in tr.indices]
    all_windows = [ds.sequences[i] for i in range(len(ds))]

    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)

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

    cache = {}

    def setup(seed):
        if seed not in cache:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train_windows),
                             size=min(CALIBRATION_SIZE, len(train_windows)),
                             replace=False)
            calib = [np.asarray(train_windows[i], dtype=np.float32) for i in idx]
            thr = est_percentile(score_all(target, calib))
            print(f"[cache] scoring all {len(all_windows)} windows for seed {seed}",
                  flush=True)
            sc = np.asarray([target.raw_score(w) for w in all_windows])
            order = np.argsort(-sc)
            probes = [np.asarray(all_windows[i], dtype=np.float32)
                      for i in order[:N_PROBES] if sc[i] > thr]
            cache[seed] = (calib, probes, sc)
        calib, probes, sc = cache[seed]
        return np.random.default_rng(seed + 7000), calib, probes, sc

    # ---------------- A3 ----------------
    print("[A3] locating the breakdown point\n", flush=True)
    a3 = {}
    for p in PASSES:
        a3[p] = {}
        for rate in RATES_A3:
            hid, fas = [], []
            for seed in SEEDS:
                rng, calib, probes, sc = setup(seed)
                cleaned, _ = filter_passes(np.stack(calib), p)
                thr = est_median_mad(
                    score_all(target, [cleaned[i] for i in range(len(cleaned))]), k_ref)
                fas.append(100.0 * float((sc > thr).mean()))
                h = 0
                for probe in probes:
                    po = poison_full(calib, probe.mean(axis=0), rate, rng)
                    cp, _ = filter_passes(np.stack(po), p)
                    s = score_all(target, [cp[i] for i in range(len(cp))])
                    if target.raw_score(probe) <= est_median_mad(s, k_ref):
                        h += 1
                hid.append(100.0 * h / len(probes))
            a3[p][rate] = {"attack": float(np.mean(hid)), "sd": float(np.std(hid)),
                           "fa": float(np.mean(fas))}
            print(f"[A3 {p} pass] {int(rate*100):>3}%  attack {np.mean(hid):5.1f} "
                  f"+/- {np.std(hid):4.1f}", flush=True)
        print(flush=True)

    # ---------------- A6 ----------------
    print("[A6] adversaries that know the pass count\n", flush=True)
    _, calib0, _, _ = setup(SEEDS[0])
    print("  feasible region left to the attacker, mean box width across")
    print("  features and timesteps, on real calibration windows:")
    for p in PASSES:
        widths = []
        for w in calib0[:40]:
            for t in range(w.shape[0]):
                lo, hi = box_after_passes(w, t, p)
                widths.append(float(np.mean(hi - lo)))
        wm = float(np.mean(widths))
        print(f"     {p} pass: {wm:.4f}" +
              ("   (collapsed: nothing can be injected)" if wm < 1e-6 else ""))
    print(flush=True)
    a6 = {}
    for p in PASSES:
        a6[p] = {}
        for label in ["full", "sub-threshold (adapted)", "white-box (adapted)"]:
            hid = []
            for seed in SEEDS:
                rng, calib, probes, sc = setup(seed)
                h = 0
                for probe in probes:
                    trig = probe.mean(axis=0)
                    if label == "full":
                        po = poison_full(calib, trig, RATE_A6, rng)
                    elif label.startswith("sub"):
                        po = poison_subthreshold(calib, trig, RATE_A6, rng, p)
                    else:
                        po = poison_whitebox(calib, trig, RATE_A6, rng, p,
                                             target, mean, std, device)
                    cp, _ = filter_passes(np.stack(po), p)
                    s = score_all(target, [cp[i] for i in range(len(cp))])
                    if target.raw_score(probe) <= est_median_mad(s, k_ref):
                        h += 1
                hid.append(100.0 * h / len(probes))
            a6[p][label] = {"attack": float(np.mean(hid)), "sd": float(np.std(hid))}
            print(f"[A6 {p} pass] {label:<24} attack {np.mean(hid):5.1f} "
                  f"+/- {np.std(hid):4.1f}", flush=True)
        print(flush=True)

    print("=" * 88)
    print("A3  WHERE THE DEFENCE BREAKS DOWN")
    print("=" * 88)
    print("%-8s" % "rate" + "".join("%-20s" % ("%d pass" % p) for p in PASSES))
    print("-" * 88)
    for rate in RATES_A3:
        row = "%-8s" % ("%d%%" % int(rate * 100))
        for p in PASSES:
            r = a3[p][rate]
            row += "%-20s" % ("%.1f +/- %.1f" % (r["attack"], r["sd"]))
        print(row)
    print("-" * 88)
    print("%-8s" % "FA %" + "".join("%-20.2f" % a3[p][RATES_A3[0]]["fa"] for p in PASSES))

    print()
    print("=" * 88)
    print("A6  ADVERSARIES THAT KNOW HOW MANY PASSES THE FILTER RUNS")
    print("=" * 88)
    print("%-26s" % "adversary" + "".join("%-20s" % ("%d pass" % p) for p in PASSES))
    print("-" * 88)
    for label in ["full", "sub-threshold (adapted)", "white-box (adapted)"]:
        row = "%-26s" % label
        for p in PASSES:
            r = a6[p][label]
            row += "%-20s" % ("%.1f +/- %.1f" % (r["attack"], r["sd"]))
        print(row)
    print("-" * 88)

    print()
    print("=" * 88)
    print("READING")
    print("=" * 88)
    for p in PASSES:
        broke = [r for r in RATES_A3 if a3[p][r]["attack"] > 20]
        if broke:
            print(f"  {p} pass: exceeds 20% attack success from "
                  f"{int(min(broke)*100)}% contamination upward")
        else:
            print(f"  {p} pass: stays below 20% at every rate tested, to "
                  f"{int(max(RATES_A3)*100)}%")
    print()
    worst6 = max(a6[2][l]["attack"] for l in a6[2])
    if worst6 < 5:
        print("  At two passes no adversary exceeds 5%, including those that")
        print("  target the final pass rather than the first.")
    else:
        print(f"  At two passes the strongest adaptive adversary reaches "
              f"{worst6:.1f}%. Report it.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "defence_limits.json", "w") as f:
        json.dump({"rates_a3": RATES_A3, "rate_a6": RATE_A6, "passes": PASSES,
                   "seeds": SEEDS, "n_probes": N_PROBES, "k_parity": k_ref,
                   "a3": {str(p): {str(k): v for k, v in d.items()}
                          for p, d in a3.items()},
                   "a6": {str(p): d for p, d in a6.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/defence_limits.json")


if __name__ == "__main__":
    main()