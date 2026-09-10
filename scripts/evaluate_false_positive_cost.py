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
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target

CALIBRATION_SIZE = 300
Z_LOCAL = 1.5
N_DRAWS = 5


def est_percentile(s, pct=99.0, **kw):
    return float(np.percentile(s, pct))


def est_trimmed(s, pct=99.0, trim=0.1, **kw):
    a = np.sort(np.asarray(s)); n = len(a); t = int(n * trim)
    a = a[t:n - t] if t > 0 else a
    return float(np.percentile(a, pct))


def est_median_mad(s, k=3.0, **kw):
    a = np.asarray(s); m = np.median(a)
    return float(m + k * 1.4826 * np.median(np.abs(a - m)))


def est_bootstrap(s, pct=99.0, B=50, rng=None, **kw):
    a = np.asarray(s); rng = rng or np.random.default_rng(0); n = len(a)
    return float(np.median([np.percentile(a[rng.integers(0, n, n)], pct) for _ in range(B)]))


ESTIMATORS = {"percentile": est_percentile, "trimmed percentile": est_trimmed,
              "median+MAD": est_median_mad, "bootstrap percentile": est_bootstrap}


def clean_local(windows, z=Z_LOCAL):
    out = []
    for w in windows:
        wc = w.copy()
        for t in range(w.shape[0]):
            o = np.delete(w, t, axis=0)
            med = np.median(o, axis=0)
            mad = np.median(np.abs(o - med), axis=0) + 1e-8
            if (np.abs(w[t] - med) / (1.4826 * mad)).max() > z:
                wc[t] = med
        out.append(wc)
    return out


def fit_k(scores, target):
    a = np.asarray(scores); m = np.median(a)
    mad = np.median(np.abs(a - m)) * 1.4826
    return 3.0 if mad <= 0 else float((target - m) / mad)


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        config = yaml.safe_load(f)
    seq_len = config["training"]["seq_len"]
    mc = {"input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
          "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
          "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"]}

    print("[setup] loading checkpoint...")
    target = load_pampos_target(REPO_ROOT, mc)
    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / config["data"]["raw_dir"], seq_len=seq_len)
    gen = torch.Generator().manual_seed(config["training"]["seed"])
    vs = max(1, int(len(ds) * config["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=gen)
    train_windows = [ds.sequences[i] for i in tr.indices]
    all_windows = [ds.sequences[i] for i in range(len(ds))]

    print(f"[setup] {len(all_windows)} windows total\n")
    print("=" * 88)
    print("FALSE-POSITIVE COST OF EACH CALIBRATION METHOD")
    print("=" * 88)
    print("All data here is BENIGN, so every flag is a false alarm.")
    print("A method that lowers attack success by flagging far more traffic is not")
    print("more secure -- it is merely more paranoid.\n")

    rows = {}
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(draw)
        idx = rng.choice(len(train_windows), size=CALIBRATION_SIZE, replace=False)
        calib = [train_windows[i] for i in idx]

        for cleaned in [False, True]:
            base = clean_local(calib) if cleaned else calib
            target.feature_mae = None
            target.calibrate(base)
            cal_scores = [target.raw_score(w) for w in base]
            all_scores = np.array([target.raw_score(w) for w in all_windows])

            p99 = est_percentile(cal_scores)
            k = fit_k(cal_scores, p99)

            for name, fn in ESTIMATORS.items():
                kw = {"k": k} if name == "median+MAD" else (
                     {"rng": np.random.default_rng(draw)} if name == "bootstrap percentile" else {})
                thr = fn(cal_scores, **kw)
                flagged = int((all_scores > thr).sum())
                label = f"{'local-clean + ' if cleaned else ''}{name}"
                rows.setdefault(label, {"thr": [], "flagged": []})
                rows[label]["thr"].append(thr)
                rows[label]["flagged"].append(flagged)

    n = len(all_windows)
    print(f"{'configuration':<36}{'threshold':<16}{'flagged':<16}{'false-alarm rate':<18}")
    print("-" * 88)
    out = {}
    baseline_rate = None
    for label, d in rows.items():
        t = np.mean(d["thr"]); f = np.mean(d["flagged"])
        rate = 100 * f / n
        if label == "percentile":
            baseline_rate = rate
        print(f"{label:<36}{t:<16.4f}{f:<16.1f}{rate:<18.2f}")
        out[label] = {"threshold_mean": float(t), "flagged_mean": float(f), "false_alarm_pct": float(rate)}

    print("-" * 88)
    if baseline_rate:
        print(f"\nRelative to the percentile baseline ({baseline_rate:.2f}% false alarms):")
        for label, v in sorted(out.items(), key=lambda kv: kv[1]["false_alarm_pct"]):
            mult = v["false_alarm_pct"] / baseline_rate if baseline_rate else 0
            note = ""
            if mult > 3:
                note = "   <-- flags far more benign traffic"
            elif mult < 0.5:
                note = "   <-- flags much less; may be missing real anomalies"
            print(f"  {label:<36}{mult:>6.2f}x{note}")

    o = REPO_ROOT / "outputs" / "results"
    o.mkdir(parents=True, exist_ok=True)
    with open(o / "false_positive_cost.json", "w") as f:
        json.dump({"n_windows": n, "n_draws": N_DRAWS, "results": out}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/false_positive_cost.json")


if __name__ == "__main__":
    main()