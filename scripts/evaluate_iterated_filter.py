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
PASSES = [1, 2, 3]
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


def constraint_box(window, t, z=Z_LOCAL):
    others = np.delete(window, t, axis=0)
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


def poison_subthreshold(windows, trigger, fraction, rng, alpha=0.9):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    locs = rng.choice(len(windows) * T,
                      size=max(1, int(fraction * len(windows) * T)), replace=False)
    for loc in locs:
        wi, t = divmod(loc, T)
        w = out[wi]
        lo, hi = constraint_box(w, t)
        med = np.median(np.delete(w, t, axis=0), axis=0)
        w[t] = np.clip(med + (trigger - med) * alpha, lo, hi)
    return out


def poison_whitebox(windows, trigger, fraction, rng, target, mean, std, device):
    out = [w.copy() for w in windows]
    T = windows[0].shape[0]
    locs = rng.choice(len(windows) * T,
                      size=max(1, int(fraction * len(windows) * T)), replace=False)
    by_window = {}
    for loc in locs:
        wi, t = divmod(loc, T)
        by_window.setdefault(wi, []).append(t)

    from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
    for wi, ts in by_window.items():
        w = out[wi]
        boxes = {t: constraint_box(w, t) for t in ts}
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
    print(f"[setup] Applying the published filter 1, 2 or 3 times.")
    print(f"[setup] Part 1 sweeps the contamination rate against the")
    print(f"[setup]   non-adaptive attacker at {N_PROBES} probes per seed.")
    print(f"[setup] Part 2 puts each pass count against the two adaptive")
    print(f"[setup]   adversaries at the paper's 15% budget.")
    print(f"[setup] {len(SEEDS)} seeds throughout.\n")

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

    _cache = {}

    def setup(seed):
        """The calibration draw, the probe set and thewindow scores depend only on
        the seed, not on the pass count or the contamination rate. Scoring all
        91,697 windows is by far the most expensive step here, so it is done
        once per seed and reused."""
        if seed not in _cache:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train_windows),
                             size=min(CALIBRATION_SIZE, len(train_windows)),
                             replace=False)
            calib = [np.asarray(train_windows[i], dtype=np.float32) for i in idx]
            clean_thr = est_percentile(score_all(target, calib))
            print(f"[cache] scoring all {len(all_windows)} windows for seed {seed}",
                  flush=True)
            sc = np.asarray([target.raw_score(w) for w in all_windows])
            order = np.argsort(-sc)
            probes = [np.asarray(all_windows[i], dtype=np.float32)
                      for i in order[:N_PROBES] if sc[i] > clean_thr]
            _cache[seed] = (calib, probes, sc)
        calib, probes, sc = _cache[seed]
        return np.random.default_rng(seed + 1000), calib, probes, sc

    # ---- part 1: contamination sweep -------------------------------------
    sweep = {}
    for p in PASSES:
        sweep[p] = {}
        for frac in FRACTIONS:
            hid, fas = [], []
            for seed in SEEDS:
                rng, calib, probes, sc = setup(seed)
                cleaned, _ = filter_passes(np.stack(calib), p)
                thr_clean = est_median_mad(
                    score_all(target, [cleaned[i] for i in range(len(cleaned))]), k_ref)
                fas.append(100.0 * float((sc > thr_clean).mean()))
                h = 0
                for probe in probes:
                    po = poison_full(calib, probe.mean(axis=0), frac, rng)
                    cp, _ = filter_passes(np.stack(po), p)
                    s = score_all(target, [cp[i] for i in range(len(cp))])
                    if target.raw_score(probe) <= est_median_mad(s, k_ref):
                        h += 1
                hid.append(100.0 * h / len(probes))
            sweep[p][frac] = {"attack": float(np.mean(hid)), "sd": float(np.std(hid)),
                              "fa": float(np.mean(fas))}
            print(f"[{p} pass] {int(frac*100):>3}%  attack {np.mean(hid):5.1f} "
                  f"+/- {np.std(hid):4.1f}   FA {np.mean(fas):5.2f}", flush=True)
        print(flush=True)

    # ---- part 2: adaptive adversaries ------------------------------------
    print("[part 2] adaptive adversaries at 15%\n", flush=True)
    adaptive = {}
    for p in PASSES:
        adaptive[p] = {}
        for label in ["full", "sub-threshold", "white-box"]:
            hid = []
            for seed in SEEDS:
                rng, calib, probes, sc = setup(seed)
                h = 0
                for probe in probes:
                    trig = probe.mean(axis=0)
                    if label == "full":
                        po = poison_full(calib, trig, 0.15, rng)
                    elif label == "sub-threshold":
                        po = poison_subthreshold(calib, trig, 0.15, rng)
                    else:
                        po = poison_whitebox(calib, trig, 0.15, rng,
                                             target, mean, std, device)
                    cp, _ = filter_passes(np.stack(po), p)
                    s = score_all(target, [cp[i] for i in range(len(cp))])
                    if target.raw_score(probe) <= est_median_mad(s, k_ref):
                        h += 1
                hid.append(100.0 * h / len(probes))
            adaptive[p][label] = {"attack": float(np.mean(hid)), "sd": float(np.std(hid))}
            print(f"[{p} pass] {label:<16} attack {np.mean(hid):5.1f} "
                  f"+/- {np.std(hid):4.1f}", flush=True)
        print(flush=True)

    print("=" * 86)
    print("ATTACK SUCCESS BY CONTAMINATION RATE AND NUMBER OF FILTER PASSES")
    print("=" * 86)
    print(f"{'rate':<8}" + "".join(f"{f'{p} pass':<20}" for p in PASSES))
    print("-" * 86)
    for frac in FRACTIONS:
        row = "%-8s" % ("%d%%" % int(frac*100))
        for p in PASSES:
            r = sweep[p][frac]
            cell = "%.1f +/- %.1f" % (r['attack'], r['sd'])
            row += "%-20s" % cell
        print(row)
    print("-" * 86)
    print(f"{'FA %':<8}" + "".join(f"{sweep[p][FRACTIONS[0]]['fa']:<20.2f}" for p in PASSES))

    print()
    print("=" * 86)
    print("ATTACK SUCCESS AGAINST ADAPTIVE ADVERSARIES, 15% BUDGET")
    print("=" * 86)
    print(f"{'adversary':<18}" + "".join(f"{f'{p} pass':<20}" for p in PASSES))
    print("-" * 86)
    for label in ["full", "sub-threshold", "white-box"]:
        row = "%-18s" % label
        for p in PASSES:
            r = adaptive[p][label]
            cell = "%.1f +/- %.1f" % (r['attack'], r['sd'])
            row += "%-20s" % cell
        print(row)
    print("-" * 86)

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "iterated_filter.json", "w") as f:
        json.dump({"fractions": FRACTIONS, "passes": PASSES, "seeds": SEEDS,
                   "n_probes": N_PROBES, "k_parity": k_ref,
                   "sweep": {str(p): {str(k): v for k, v in d.items()}
                             for p, d in sweep.items()},
                   "adaptive": {str(p): d for p, d in adaptive.items()}},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/iterated_filter.json")


if __name__ == "__main__":
    main()