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
from src.attacks.adversarial_ml_attacks.backdoor_attack import create_backdoor_attack

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]
CALIBRATION_SIZE = 300
N_PROBES = 20
N_SEEDS = 3
Z_LOCAL = 1.5


def est_percentile(s, pct=99.0, **kw):
    return float(np.percentile(s, pct))


def est_median_mad(s, k=3.0, **kw):
    a = np.asarray(s); m = np.median(a)
    return float(m + k * 1.4826 * np.median(np.abs(a - m)))


def fit_k(scores, target):
    a = np.asarray(scores); m = np.median(a)
    mad = np.median(np.abs(a - m)) * 1.4826
    return 3.0 if mad <= 0 else float((target - m) / mad)


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


def set_scale(target, windows):
    """Compute feature_mae (the per-feature error scale) from these windows."""
    target.feature_mae = None
    target.calibrate(windows)
    return target.feature_mae.clone()


def score_with(target, scale, windows):
    target.feature_mae = scale
    return [target.raw_score(w) for w in windows]


def evaluate(target, calib, all_windows, probes, decouple, estimator, k_ref, seed):
    """
    decouple=False -- feature_mae AND threshold both from the cleaned data (current)
    decouple=True  -- feature_mae from the ORIGINAL data, threshold from cleaned
    """
    cleaned = clean_local(calib)
    scale = set_scale(target, calib if decouple else cleaned)
    cal_scores = score_with(target, scale, cleaned)

    kw = {"k": k_ref} if estimator is est_median_mad else {}
    thr = estimator(cal_scores, **kw)

    all_scores = np.array(score_with(target, scale, all_windows))
    false_alarms = int((all_scores > thr).sum())
    probes_kept = sum(1 for p in probes if target.raw_score(p) > thr)

    hidden = 0
    for probe in probes:
        trig = probe.mean(axis=0)
        atk = create_backdoor_attack(
            trigger_pattern=dict(zip(FEATURE_NAMES, trig.tolist())),
            fraction_poisoned=0.15)
        poisoned = atk.poison(calib, seed=seed)
        p_cleaned = clean_local(poisoned)
        p_scale = set_scale(target, poisoned if decouple else p_cleaned)
        p_scores = score_with(target, p_scale, p_cleaned)
        p_thr = estimator(p_scores, **kw)
        target.feature_mae = p_scale
        if target.raw_score(probe) <= p_thr:
            hidden += 1

    return {"threshold": thr, "false_alarms": false_alarms,
            "false_alarm_pct": 100 * false_alarms / len(all_windows),
            "probes_kept": probes_kept, "hidden": hidden,
            "attack_success": hidden / len(probes)}


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

    print("=" * 92)
    print("DOES DECOUPLING feature_mae REMOVE THE FALSE-ALARM COST OF CLEANING?")
    print("=" * 92)
    print("Cleaning removes variability from the calibration data, which shrinks feature_mae.")
    print("Scores are residuals DIVIDED by feature_mae, so every score inflates and more")
    print("benign windows cross the threshold. Fix under test: take feature_mae from the")
    print("ORIGINAL data and only the threshold from the cleaned data.\n")

    configs = [
        ("no defence (percentile)",            None,  est_percentile),
        ("clean, coupled + percentile",        False, est_percentile),
        ("clean, DECOUPLED + percentile",      True,  est_percentile),
        ("clean, coupled + median+MAD",        False, est_median_mad),
        ("clean, DECOUPLED + median+MAD",      True,  est_median_mad),
    ]
    acc = {name: {"fa": [], "as": [], "kept": [], "thr": []} for name, _, _ in configs}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_windows), size=CALIBRATION_SIZE, replace=False)
        calib = [train_windows[i] for i in idx]

        scale0 = set_scale(target, calib)
        base_scores = score_with(target, scale0, calib)
        base_thr = est_percentile(base_scores)
        k_ref = fit_k(base_scores, base_thr)

        all_scores = np.array(score_with(target, scale0, all_windows))
        flagged = sorted([(w, s) for w, s in zip(all_windows, all_scores) if s > base_thr],
                         key=lambda p: p[1], reverse=True)
        k = min(N_PROBES, len(flagged))
        if k == 0:
            continue
        probes = [w for w, _ in flagged[:k]]
        print(f"[seed {seed}] baseline threshold {base_thr:.3f}, {len(flagged)} flagged, {k} probes")

        for name, decouple, est in configs:
            if decouple is None:
                target.feature_mae = scale0
                fa = int((all_scores > base_thr).sum())
                r = {"threshold": base_thr, "false_alarms": fa,
                     "false_alarm_pct": 100 * fa / len(all_windows),
                     "probes_kept": k, "hidden": k, "attack_success": 1.0}
            else:
                r = evaluate(target, calib, all_windows, probes, decouple, est, k_ref, seed)
            acc[name]["fa"].append(r["false_alarm_pct"])
            acc[name]["as"].append(r["attack_success"])
            acc[name]["kept"].append(r["probes_kept"] / k)
            acc[name]["thr"].append(r["threshold"])
            print(f"    {name:<34} thr {r['threshold']:.2f}  "
                  f"false alarms {r['false_alarm_pct']:.2f}%  attack {100*r['attack_success']:.0f}%")
        print()

    print("=" * 92)
    print(f"{'configuration':<34}{'threshold':<12}{'false alarms':<18}{'attack success':<20}{'probes kept':<12}")
    print("-" * 92)
    out = {}
    for name, _, _ in configs:
        d = acc[name]
        if not d["fa"]:
            continue
        print(f"{name:<34}{np.mean(d['thr']):<12.2f}{np.mean(d['fa']):<18.2f}"
              f"{np.mean(d['as'])*100:>6.1f}% +/- {np.std(d['as'])*100:<10.1f}{np.mean(d['kept'])*100:<12.0f}")
        out[name] = {"threshold": float(np.mean(d["thr"])),
                     "false_alarm_pct": float(np.mean(d["fa"])),
                     "attack_success_mean": float(np.mean(d["as"])),
                     "attack_success_std": float(np.std(d["as"])),
                     "probes_kept": float(np.mean(d["kept"]))}
    print("-" * 92)
    print("\nThe fix succeeds if a DECOUPLED row keeps attack success low while bringing")
    print("false alarms back near the no-defence baseline.")

    o = REPO_ROOT / "outputs" / "results"
    o.mkdir(parents=True, exist_ok=True)
    with open(o / "decoupled_scale_comparison.json", "w") as f:
        json.dump({"n_seeds": N_SEEDS, "results": out}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/decoupled_scale_comparison.json")


if __name__ == "__main__":
    main()