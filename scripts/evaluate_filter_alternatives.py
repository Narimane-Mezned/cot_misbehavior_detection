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
FRACTION_POISONED = 0.15
N_PROBES = 20
SEEDS = [0, 1, 2]
K_PARITY_PATH = "outputs/results/calibration_method_comparison.json"


def filter_none(W):
    return W.copy(), 0.0


def filter_within_window_median(W, z=Z_LOCAL):
    """The filter used in the paper: replace any timestep whose within-window
    z-score exceeds z with the within-window median."""
    n, T, _ = W.shape
    out = W.copy()
    replaced = 0
    for t in range(T):
        others = np.delete(W, t, axis=1)
        med = np.median(others, axis=1)
        mad = np.median(np.abs(others - med[:, None, :]), axis=1) + 1e-8
        hit = (np.abs(W[:, t, :] - med) / (1.4826 * mad)).max(axis=1) > z
        out[hit, t, :] = med[hit]
        replaced += int(hit.sum())
    return out, replaced / (n * T)


def filter_hampel(W, z=Z_LOCAL, half=2):
    """Hampel filter: judge each timestep against a local window of its
    +/- half neighbours rather than the whole sequence, and replace only the
    offending feature rather than the whole timestep."""
    n, T, F = W.shape
    out = W.copy()
    touched = np.zeros((n, T), dtype=bool)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        idx = [u for u in range(lo, hi) if u != t]
        if not idx:
            continue
        loc = W[:, idx, :]
        med = np.median(loc, axis=1)
        mad = np.median(np.abs(loc - med[:, None, :]), axis=1) + 1e-8
        dev = np.abs(W[:, t, :] - med) / (1.4826 * mad)
        hit = dev > z
        out[:, t, :][hit] = med[hit]
        touched[:, t] = hit.any(axis=1)
    return out, float(touched.mean())


def filter_median_of_means(W, z=Z_LOCAL, blocks=5):
    """Temporal median-of-means: split the window into blocks, take the mean
    of each, and judge timesteps against the median of those block means."""
    n, T, _ = W.shape
    out = W.copy()
    edges = np.array_split(np.arange(T), blocks)
    means = np.stack([W[:, e, :].mean(axis=1) for e in edges], axis=1)
    med = np.median(means, axis=1)
    mad = np.median(np.abs(means - med[:, None, :]), axis=1) + 1e-8
    replaced = 0
    for t in range(T):
        hit = (np.abs(W[:, t, :] - med) / (1.4826 * mad)).max(axis=1) > z
        out[hit, t, :] = med[hit]
        replaced += int(hit.sum())
    return out, replaced / (n * T)


FILTERS = [
    ("none", filter_none),
    ("within-window median (ours)", filter_within_window_median),
    ("Hampel, half-width 2", filter_hampel),
    ("temporal median-of-means", filter_median_of_means),
]


def est_percentile(scores, pct=99.0):
    return float(np.percentile(scores, pct))


def est_median_mad(scores, k):
    s = np.asarray(scores)
    med = np.median(s)
    return float(med + k * 1.4826 * np.median(np.abs(s - med)))


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


def run_seed(target, train_windows, all_windows, seed, k_ref):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows),
                     size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
    calib = [np.asarray(train_windows[i], dtype=np.float32) for i in idx]

    clean_scores = score_all(target, calib)
    clean_thr = est_percentile(clean_scores)

    scores = [target.raw_score(w) for w in all_windows]
    flagged = sorted([(w, s) for w, s in zip(all_windows, scores) if s > clean_thr],
                     key=lambda p: p[1], reverse=True)
    if not flagged:
        return None
    probes = [np.asarray(w, dtype=np.float32) for w, _ in flagged[:N_PROBES]]
    all_scores = np.asarray(scores)

    out = {}
    for name, fn in FILTERS:
        cleaned_calib, replaced = fn(np.stack(calib))
        cleaned_list = [cleaned_calib[i] for i in range(len(calib))]
        sc_clean = score_all(target, cleaned_list)
        thr_clean = est_median_mad(sc_clean, k_ref)
        fa = 100.0 * float((all_scores > thr_clean).mean())
        utility = 100.0 * float(np.mean([target.raw_score(p) > thr_clean for p in probes]))

        hidden = 0
        for probe in probes:
            poisoned = poison(calib, probe.mean(axis=0), FRACTION_POISONED, rng)
            cp, _ = fn(np.stack(poisoned))
            sc = score_all(target, [cp[i] for i in range(len(poisoned))])
            if target.raw_score(probe) <= est_median_mad(sc, k_ref):
                hidden += 1

        out[name] = {"attack_success": 100.0 * hidden / len(probes),
                     "false_alarm": fa, "utility": utility,
                     "replaced": 100.0 * replaced}
    return out


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kp = REPO_ROOT / K_PARITY_PATH
    if not kp.exists():
        print(f"[abort] {K_PARITY_PATH} not found; run evaluate_calibration_methods.py first")
        return
    k_ref = float(json.load(open(kp))["k_parity"])

    print(f"[setup] device {device}, k for median+MAD = {k_ref:.4f}")
    print(f"[setup] four filters, all followed by median+MAD at the same k")
    print(f"[setup] {CALIBRATION_SIZE}-window calibration, {int(FRACTION_POISONED*100)}% poisoning,")
    print(f"[setup] {len(SEEDS)} seeds x {N_PROBES} probes")
    print(f"[setup] the question: can a lighter filter buy the same protection")
    print(f"[setup] at a lower false-alarm cost?\n")

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

    trials = []
    for seed in SEEDS:
        r = run_seed(target, train_windows, all_windows, seed, k_ref)
        if r:
            trials.append(r)
            for name, _ in FILTERS:
                c = r[name]
                print(f"[seed {seed}] {name:<30} attack {c['attack_success']:5.1f}  "
                      f"FA {c['false_alarm']:5.2f}  replaced {c['replaced']:5.1f}", flush=True)
            print(flush=True)

    if not trials:
        print("[abort] no usable trials")
        return

    print("=" * 92)
    print("ALTERNATIVE FILTERS ON THE PROTECTION / COST FRONTIER")
    print("=" * 92)
    print("Every row uses median+MAD thresholding at the same k. Only the")
    print("pre-calibration filter differs.")
    print()
    print(f"{'filter':<32}{'attack success':<20}{'benign FA %':<14}"
          f"{'utility %':<12}{'replaced %'}")
    print("-" * 92)

    table = {}
    for name, _ in FILTERS:
        a = [t[name]["attack_success"] for t in trials]
        fa = [t[name]["false_alarm"] for t in trials]
        ut = [t[name]["utility"] for t in trials]
        rp = [t[name]["replaced"] for t in trials]
        table[name] = {"attack_success": float(np.mean(a)), "attack_sd": float(np.std(a)),
                       "false_alarm": float(np.mean(fa)), "utility": float(np.mean(ut)),
                       "replaced": float(np.mean(rp))}
        print(f"{name:<32}{f'{np.mean(a):.1f} +/- {np.std(a):.1f}':<20}"
              f"{np.mean(fa):<14.2f}{np.mean(ut):<12.1f}{np.mean(rp):.1f}")
    print("-" * 92)

    print()
    print("=" * 92)
    print("READING")
    print("=" * 92)
    ours = table["within-window median (ours)"]
    alts = {k: v for k, v in table.items()
            if k not in ("none", "within-window median (ours)")}
    better = {k: v for k, v in alts.items()
              if v["attack_success"] <= ours["attack_success"] + 5
              and v["false_alarm"] < ours["false_alarm"]}
    print(f"  ours: {ours['attack_success']:.1f}% attack success at "
          f"{ours['false_alarm']:.2f}% false alarms, {ours['replaced']:.1f}% replaced")
    for k, v in alts.items():
        print(f"  {k}: {v['attack_success']:.1f}% at {v['false_alarm']:.2f}%, "
              f"{v['replaced']:.1f}% replaced")
    print()
    if better:
        k = min(better, key=lambda x: better[x]["false_alarm"])
        print(f"  {k} attains comparable protection at a lower false-alarm cost.")
        print(f"  The filter we adopted is not on the frontier, and the paper should")
        print(f"  say so.")
    else:
        print("  No lighter filter tested attains comparable protection at a lower")
        print("  cost. The frontier we report is not obviously improvable within")
        print("  this family, though that is a statement about these three filters")
        print("  and not about all possible ones.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "filter_alternatives.json", "w") as f:
        json.dump({"z_local": Z_LOCAL, "fraction": FRACTION_POISONED,
                   "calibration_size": CALIBRATION_SIZE, "seeds": SEEDS,
                   "n_probes": N_PROBES, "k_parity": k_ref, "table": table},
                  f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/filter_alternatives.json")


if __name__ == "__main__":
    main()