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
Z_THRESHOLD = 1.5


def clean_windows_via_local_consistency(windows, z_threshold=Z_THRESHOLD):
    cleaned = []
    for w in windows:
        wc = w.copy()
        for t in range(w.shape[0]):
            others = np.delete(w, t, axis=0)
            med = np.median(others, axis=0)
            mad = np.median(np.abs(others - med), axis=0) + 1e-8
            z = np.abs(w[t] - med) / (1.4826 * mad)
            if z.max() > z_threshold:
                wc[t] = med
        cleaned.append(wc)
    return cleaned


def build_trigger(level, probe, other_flagged, benign_pool, rng):
    """
    full        -- attacker knows the exact target window (current, worst case)
    category    -- attacker knows the KIND of anomaly, not the instance:
                   trigger built from other flagged windows, target excluded
    distribution-- attacker knows only the benign traffic distribution
    """
    if level == "full":
        return probe.mean(axis=0)

    if level == "category":
        if len(other_flagged) == 0:
            return None
        k = min(5, len(other_flagged))
        idx = rng.choice(len(other_flagged), size=k, replace=False)
        return np.mean([other_flagged[i].mean(axis=0) for i in idx], axis=0)

    if level == "distribution":
        k = min(5, len(benign_pool))
        idx = rng.choice(len(benign_pool), size=k, replace=False)
        return np.mean([benign_pool[i].mean(axis=0) for i in idx], axis=0)

    raise ValueError(level)


def run_trial(target, train_windows, all_windows, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), size=min(CALIBRATION_SIZE, len(train_windows)), replace=False)
    calibration_windows = [train_windows[i] for i in idx]

    target.feature_mae = None
    target.calibrate(calibration_windows)
    clean_threshold = target.threshold

    scores = [target.raw_score(w) for w in all_windows]
    flagged = [(w, s) for w, s in zip(all_windows, scores) if s > clean_threshold]
    flagged.sort(key=lambda p: p[1], reverse=True)
    k = min(N_PROBES, len(flagged))
    if k == 0:
        return None
    probes = [w for w, s in flagged[:k]]
    all_flagged = [w for w, s in flagged]

    out = {"seed": seed, "clean_threshold": clean_threshold,
           "n_flagged": len(flagged), "n_probes": k, "levels": {}}

    for level in ["full", "category", "distribution"]:
        hidden_undefended = 0
        hidden_defended = 0
        skipped = 0

        for pi, probe in enumerate(probes):
            others = [w for j, w in enumerate(all_flagged) if not np.array_equal(w, probe)]
            trigger = build_trigger(level, probe, others, calibration_windows, rng)
            if trigger is None:
                skipped += 1
                continue

            attack = create_backdoor_attack(
                trigger_pattern=dict(zip(FEATURE_NAMES, trigger.tolist())),
                fraction_poisoned=0.15,
            )
            poisoned = attack.poison(calibration_windows, seed=seed)

            target.feature_mae = None
            target.calibrate(poisoned)
            if target.raw_score(probe) <= target.threshold:
                hidden_undefended += 1

            cleaned = clean_windows_via_local_consistency(poisoned)
            target.feature_mae = None
            target.calibrate(cleaned)
            if target.raw_score(probe) <= target.threshold:
                hidden_defended += 1

        n = k - skipped
        out["levels"][level] = {
            "n_tested": n,
            "hidden_undefended": hidden_undefended,
            "hidden_defended": hidden_defended,
            "rate_undefended": hidden_undefended / n if n else None,
            "rate_defended": hidden_defended / n if n else None,
        }

    target.feature_mae = None
    target.calibrate(calibration_windows)
    return out


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint...")
    target = load_pampos_target(REPO_ROOT, model_config)

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    gen = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, _ = random_split(full_dataset, [train_size, val_size], generator=gen)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]
    all_windows = [full_dataset.sequences[i] for i in range(len(full_dataset))]

    print(f"[setup] {len(train_windows)} train / {len(all_windows)} total windows")
    print(f"[setup] {N_SEEDS} seeds x 3 knowledge levels x {N_PROBES} probes x 2 defence conditions")
    print(f"[setup] Local-consistency defence at z={Z_THRESHOLD}. This is heavy -- allow 20-40 min.\n")

    trials = []
    for seed in range(N_SEEDS):
        print(f"[seed {seed}] running...")
        r = run_trial(target, train_windows, all_windows, seed)
        if r is None:
            print(f"[seed {seed}] no flagged windows, skipped")
            continue
        trials.append(r)
        print(f"[seed {seed}] threshold={r['clean_threshold']:.3f}, flagged={r['n_flagged']}")
        for lvl in ["full", "category", "distribution"]:
            d = r["levels"][lvl]
            print(f"    {lvl:<13} undefended {d['hidden_undefended']}/{d['n_tested']}"
                  f"   defended {d['hidden_defended']}/{d['n_tested']}")
        print()

    if not trials:
        print("[error] no usable trials")
        return

    print("=" * 84)
    print("THREAT MODEL SPECTRUM -- backdoor success by attacker knowledge (mean +/- std)")
    print("=" * 84)
    print(f"{'attacker knowledge':<28}{'no defence':<24}{'with local-consistency':<24}")
    print("-" * 84)

    summary = {}
    labels = {"full": "exact target window",
              "category": "anomaly category only",
              "distribution": "benign distribution only"}
    for lvl in ["full", "category", "distribution"]:
        u = [t["levels"][lvl]["rate_undefended"] for t in trials if t["levels"][lvl]["rate_undefended"] is not None]
        d = [t["levels"][lvl]["rate_defended"] for t in trials if t["levels"][lvl]["rate_defended"] is not None]
        if not u:
            print(f"{labels[lvl]:<28}no data")
            continue
        print(f"{labels[lvl]:<28}{np.mean(u)*100:>6.1f}% +/- {np.std(u)*100:<11.1f}"
              f"{np.mean(d)*100:>6.1f}% +/- {np.std(d)*100:<11.1f}")
        summary[lvl] = {
            "label": labels[lvl],
            "undefended_mean": float(np.mean(u)), "undefended_std": float(np.std(u)),
            "defended_mean": float(np.mean(d)), "defended_std": float(np.std(d)),
        }

    print("-" * 84)
    print("\nInterpretation: 'exact target window' is the worst case previously reported.")
    print("Lower rows are weaker, more realistic adversaries. A steep drop across rows means")
    print("the vulnerability depends heavily on attacker foreknowledge, which is worth stating")
    print("explicitly rather than reporting only the worst case.")

    out_dir = REPO_ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "threat_model_spectrum.json", "w") as f:
        json.dump({"n_seeds": len(trials), "z_threshold": Z_THRESHOLD,
                   "summary": summary, "trials": trials}, f, indent=2, default=str)
    print(f"\n[done] saved to outputs/results/threat_model_spectrum.json")


if __name__ == "__main__":
    main()