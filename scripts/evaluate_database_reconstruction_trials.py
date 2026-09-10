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
from src.attacks.adversarial_ml_attacks.database_reconstruction import (
    DatabaseReconstructionAttack, DatabaseTargetModel,
)

N_TRIALS = 30
POOL_SIZE = 800
PER_CLASS = 20
CALIBRATION_SIZE = 300


def empirical_bounds(windows, seq_len, n_features, lo=2.0, hi=98.0, pad=1.1):
    stacked = np.concatenate([w.reshape(-1, n_features) for w in windows], axis=0)
    a = np.percentile(stacked, lo, axis=0)
    b = np.percentile(stacked, hi, axis=0)
    r = b - a
    a = a - r * (pad - 1) / 2
    b = b + r * (pad - 1) / 2
    return np.tile(np.stack([a, b], axis=1), (seq_len, 1))


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        config = yaml.safe_load(f)
    seq_len = config["training"]["seq_len"]
    n_features = config["model"]["input_dim"]
    mc = {"input_dim": n_features, "d_model": config["model"]["d_model"],
          "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
          "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"]}

    print("[setup] loading checkpoint...")
    target = load_pampos_target(REPO_ROOT, mc)
    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / config["data"]["raw_dir"], seq_len=seq_len)
    gen = torch.Generator().manual_seed(config["training"]["seed"])
    vs = max(1, int(len(ds) * config["training"]["val_fraction"]))
    tr, _ = random_split(ds, [len(ds) - vs, vs], generator=gen)
    train_windows = [ds.sequences[i] for i in tr.indices]

    target.feature_mae = None
    target.calibrate(train_windows[:CALIBRATION_SIZE])
    print(f"[setup] threshold {target.threshold:.4f}")

    pool = train_windows[:min(POOL_SIZE, len(train_windows))]
    scores = [target.raw_score(w) for w in pool]
    flagged = [i for i, s in enumerate(scores) if s > target.threshold]
    unflagged = [i for i, s in enumerate(scores) if s <= target.threshold]
    per = min(PER_CLASS, len(flagged), len(unflagged))
    print(f"[setup] pool {len(pool)} windows -> {len(flagged)} flagged, {len(unflagged)} unflagged")
    print(f"[setup] using {per} per class, {N_TRIALS} independent trials\n")

    if per == 0:
        print("[error] need both classes present")
        return

    bounds = empirical_bounds(train_windows[:300], seq_len, n_features)
    input_dim = seq_len * n_features

    correct = 0
    errors = []
    per_class_correct = {0.0: 0, 1.0: 0}
    per_class_total = {0.0: 0, 1.0: 0}

    print("running trials...")
    for t in range(N_TRIALS):
        rng = np.random.default_rng(t)
        fi = rng.choice(flagged, size=per, replace=False)
        ui = rng.choice(unflagged, size=per, replace=False)
        sel = list(fi) + list(ui)

        X = np.array([pool[i].flatten() for i in sel])
        y = np.array([1.0 if scores[i] > target.threshold else 0.0 for i in sel])

        missing = int(rng.integers(0, len(sel)))
        true_row = X[missing].copy()
        true_label = y[missing]
        known_X = np.delete(X, missing, axis=0)
        known_y = np.delete(y, missing, axis=0)

        surrogate = DatabaseTargetModel(input_dim=input_dim)
        surrogate.fit(X, y)

        atk = DatabaseReconstructionAttack(
            target_factory=lambda: DatabaseTargetModel(input_dim=input_dim),
            input_bounds=bounds, max_iterations=80)
        r = atk.reconstruct(surrogate, known_X, known_y)

        ok = (r["candidate_label"] == true_label)
        correct += int(ok)
        per_class_total[true_label] += 1
        per_class_correct[true_label] += int(ok)
        errors.append(float(np.linalg.norm(np.array(r["reconstructed_features"]) - true_row)))

        if (t + 1) % 10 == 0:
            print(f"   {t+1}/{N_TRIALS} trials, label accuracy so far {correct/(t+1):.3f}")

    acc = correct / N_TRIALS
    se = np.sqrt(0.25 / N_TRIALS)
    z = (acc - 0.5) / se if se > 0 else 0.0

    print()
    print("=" * 76)
    print("DATABASE RECONSTRUCTION -- DOES LABEL RECOVERY BEAT CHANCE?")
    print("=" * 76)
    print(f"trials                        {N_TRIALS}")
    print(f"label recovery accuracy       {acc:.3f}")
    print(f"chance baseline               0.500  (balanced classes)")
    print(f"standard error                {se:.3f}")
    print(f"z-score vs chance             {z:+.2f}")
    print()
    print(f"  when the missing row was FLAGGED   : {per_class_correct[1.0]}/{per_class_total[1.0]}")
    print(f"  when the missing row was UNFLAGGED : {per_class_correct[0.0]}/{per_class_total[0.0]}")
    print()
    print(f"feature reconstruction L2 error   mean {np.mean(errors):.1f}  "
          f"min {np.min(errors):.1f}  max {np.max(errors):.1f}")

    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    if abs(z) < 1.96:
        print("Label recovery is NOT distinguishable from chance (|z| < 1.96).")
        print("The single-trial 'label recovered correctly' previously reported was one")
        print("coin flip. There is no measurable database-reconstruction leak to defend.")
    elif z >= 1.96:
        print("Label recovery IS above chance -- a real, if narrow, leak.")
        print("Note that the label is simply whether the window exceeds the threshold,")
        print("which any party with query access obtains by definition. Whether this")
        print("constitutes a meaningful disclosure is a threat-model question.")
    else:
        print("Label recovery is BELOW chance, which indicates the attack is")
        print("anti-correlated with the truth -- effectively no usable signal.")

    o = REPO_ROOT / "outputs" / "results"
    o.mkdir(parents=True, exist_ok=True)
    with open(o / "database_reconstruction_trials.json", "w") as f:
        json.dump({"n_trials": N_TRIALS, "label_accuracy": acc, "z_score": float(z),
                   "per_class_correct": {str(k): v for k, v in per_class_correct.items()},
                   "per_class_total": {str(k): v for k, v in per_class_total.items()},
                   "l2_errors": {"mean": float(np.mean(errors)),
                                 "min": float(np.min(errors)),
                                 "max": float(np.max(errors))}}, f, indent=2)
    print(f"\n[done] saved to outputs/results/database_reconstruction_trials.json")


if __name__ == "__main__":
    main()
    