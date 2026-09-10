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
N_SEEDS = 3
CALIBRATION_SIZE = 300
N_PROBES = 20
Z_LOCAL = 1.5
N_BOOTSTRAP = 50
N_STABILITY_DRAWS = 15


# ----------------------------------------------------------------------
# threshold estimators -- all operate on an already-computed score array,
# so they are cheap; the expensive part is scoring the windows once.
# ----------------------------------------------------------------------

def est_percentile(scores, pct=99.0, **kw):
    return float(np.percentile(scores, pct))


def est_trimmed(scores, pct=99.0, trim=0.1, **kw):
    s = np.sort(np.asarray(scores))
    n = len(s)
    t = int(n * trim)
    s = s[t:n - t] if t > 0 else s
    return float(np.percentile(s, pct))


def est_median_mad(scores, k=3.0, **kw):
    s = np.asarray(scores)
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    return float(med + k * 1.4826 * mad)


def est_bootstrap(scores, pct=99.0, B=N_BOOTSTRAP, rng=None, **kw):
    """Resample the calibration scores B times, take a threshold from each,
    and use the median. Averaging over subsamples both reduces the variance
    of the estimate and dilutes the influence of any injected cluster."""
    s = np.asarray(scores)
    rng = rng or np.random.default_rng(0)
    n = len(s)
    ths = [np.percentile(s[rng.integers(0, n, n)], pct) for _ in range(B)]
    return float(np.median(ths))


ESTIMATORS = {
    "percentile": est_percentile,
    "trimmed percentile": est_trimmed,
    "median+MAD": est_median_mad,
    "bootstrap percentile": est_bootstrap,
}


def fit_k_for_parity(scores, target):
    s = np.asarray(scores)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826
    return 3.0 if mad <= 0 else float((target - med) / mad)


def clean_local(windows, z=Z_LOCAL):
    out = []
    for w in windows:
        wc = w.copy()
        for t in range(w.shape[0]):
            others = np.delete(w, t, axis=0)
            med = np.median(others, axis=0)
            mad = np.median(np.abs(others - med), axis=0) + 1e-8
            if (np.abs(w[t] - med) / (1.4826 * mad)).max() > z:
                wc[t] = med
        out.append(wc)
    return out


def score_all(target, windows):
    target.feature_mae = None
    target.calibrate(windows)
    return [target.raw_score(w) for w in windows]


# ----------------------------------------------------------------------
# PART A -- stability: how much does the threshold move between draws?
# ----------------------------------------------------------------------

def stability_analysis(target, train_windows):
    print("=" * 86)
    print("PART A -- CALIBRATION STABILITY (no attack involved)")
    print("=" * 86)
    print(f"Drawing {N_STABILITY_DRAWS} independent calibration sets of {CALIBRATION_SIZE} windows.")
    print("A lower spread means the detector's operating point is reproducible.\n")

    per_est = {name: [] for name in ESTIMATORS}
    k_ref = None

    for d in range(N_STABILITY_DRAWS):
        rng = np.random.default_rng(1000 + d)
        idx = rng.choice(len(train_windows), size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
        calib = [train_windows[i] for i in idx]
        scores = score_all(target, calib)

        p99 = est_percentile(scores)
        if k_ref is None:
            k_ref = fit_k_for_parity(scores, p99)

        for name, fn in ESTIMATORS.items():
            kw = {"k": k_ref} if name == "median+MAD" else {}
            if name == "bootstrap percentile":
                kw = {"rng": np.random.default_rng(d)}
            per_est[name].append(fn(scores, **kw))

    print(f"{'estimator':<26}{'mean':<12}{'std':<12}{'min':<12}{'max':<12}{'CV':<10}")
    print("-" * 86)
    out = {}
    for name, vals in per_est.items():
        v = np.array(vals)
        cv = v.std() / v.mean() if v.mean() else 0
        print(f"{name:<26}{v.mean():<12.4f}{v.std():<12.4f}{v.min():<12.4f}{v.max():<12.4f}{cv:<10.4f}")
        out[name] = {"mean": float(v.mean()), "std": float(v.std()),
                     "min": float(v.min()), "max": float(v.max()), "cv": float(cv)}
    print("-" * 86)
    print("CV = std/mean. Lower is more stable.\n")
    return out, k_ref


# ----------------------------------------------------------------------
# PART B -- poisoning resistance
# ----------------------------------------------------------------------

def poisoning_analysis(target, train_windows, all_windows, k_ref):
    print("=" * 86)
    print("PART B -- RESISTANCE TO CALIBRATION POISONING")
    print("=" * 86)

    configs = []
    for cleaned in [False, True]:
        for name in ESTIMATORS:
            label = f"{'local-clean + ' if cleaned else ''}{name}"
            configs.append((label, cleaned, name))

    totals = {label: [] for label, _, _ in configs}
    utility = {label: [] for label, _, _ in configs}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_windows), size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
        calib = [train_windows[i] for i in idx]

        base_scores = score_all(target, calib)
        clean_thr = est_percentile(base_scores)

        scores_all = [target.raw_score(w) for w in all_windows]
        flagged = sorted([(w, s) for w, s in zip(all_windows, scores_all) if s > clean_thr],
                         key=lambda p: p[1], reverse=True)
        k = min(N_PROBES, len(flagged))
        if k == 0:
            continue
        probes = [w for w, _ in flagged[:k]]
        print(f"[seed {seed}] threshold {clean_thr:.3f}, flagged {len(flagged)}, probes {k}")

        # utility on clean calibration, per config
        cached_clean = {}
        for cleaned in [False, True]:
            base = clean_local(calib) if cleaned else calib
            cached_clean[cleaned] = score_all(target, base)
        for label, cleaned, est in configs:
            kw = {"k": k_ref} if est == "median+MAD" else ({"rng": np.random.default_rng(seed)} if est == "bootstrap percentile" else {})
            thr = ESTIMATORS[est](cached_clean[cleaned], **kw)
            kept = sum(1 for p in probes if target.raw_score(p) > thr)
            utility[label].append(kept / k)

        # attack: score once per (probe, cleaning) then apply every estimator
        hidden = {label: 0 for label, _, _ in configs}
        for probe in probes:
            trig = probe.mean(axis=0)
            atk = create_backdoor_attack(
                trigger_pattern=dict(zip(FEATURE_NAMES, trig.tolist())),
                fraction_poisoned=0.15)
            poisoned = atk.poison(calib, seed=seed)

            for cleaned in [False, True]:
                base = clean_local(poisoned) if cleaned else poisoned
                sc = score_all(target, base)
                probe_score = target.raw_score(probe)
                for label, c, est in configs:
                    if c != cleaned:
                        continue
                    kw = {"k": k_ref} if est == "median+MAD" else ({"rng": np.random.default_rng(seed)} if est == "bootstrap percentile" else {})
                    if probe_score <= ESTIMATORS[est](sc, **kw):
                        hidden[label] += 1

        for label in hidden:
            totals[label].append(hidden[label] / k)
        print(f"           " + "  ".join(f"{l.split('+ ')[-1][:9]}={hidden[l]}" for l, _, _ in configs[:4]))

    print()
    print(f"{'configuration':<34}{'attack success':<26}{'utility retained':<20}")
    print("-" * 86)
    summary = {}
    for label, _, _ in configs:
        if not totals[label]:
            continue
        a = np.array(totals[label]); u = np.array(utility[label])
        print(f"{label:<34}{a.mean()*100:>6.1f}% +/- {a.std()*100:<14.1f}{u.mean()*100:>6.1f}%")
        summary[label] = {"attack_success_mean": float(a.mean()),
                          "attack_success_std": float(a.std()),
                          "utility_mean": float(u.mean())}
    print("-" * 86)
    print("\nUtility = share of genuinely anomalous probes still flagged on CLEAN calibration.")
    print("A configuration that lowers attack success by going blind is not a defence.")

    viable = {k: v for k, v in summary.items() if v["utility_mean"] >= 0.9}
    if viable:
        best = min(viable.items(), key=lambda kv: kv[1]["attack_success_mean"])
        print(f"\nBest with utility >= 90%: {best[0]} -- "
              f"{best[1]['attack_success_mean']*100:.1f}% attack success "
              f"(utility {best[1]['utility_mean']*100:.1f}%)")
    else:
        print("\nNo configuration kept >= 90% utility.")
    return summary


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
    print(f"[setup] {len(train_windows)} train / {len(all_windows)} total\n")

    stab, k_ref = stability_analysis(target, train_windows)
    pois = poisoning_analysis(target, train_windows, all_windows, k_ref)

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibration_method_comparison.json", "w") as f:
        json.dump({"stability": stab, "poisoning": pois,
                   "k_parity": k_ref, "n_seeds": N_SEEDS,
                   "n_bootstrap": N_BOOTSTRAP}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/calibration_method_comparison.json")


if __name__ == "__main__":
    main()