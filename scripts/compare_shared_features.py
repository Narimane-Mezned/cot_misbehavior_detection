import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset

SHARED = [0, 1, 2, 3, 4]
SHARED_NAMES = ["x", "y", "vx", "vy", "yaw"]
ALL_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]


def load_model(checkpoint_path, input_dim, config, device):
    model = PAMPOS(
        input_dim=input_dim,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("val_loss")


@torch.no_grad()
def shared_feature_errors(model, windows, mean, std, keep, device):
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32)
    keep_t = torch.tensor(keep, dtype=torch.long)
    shared_t = torch.tensor(SHARED, dtype=torch.long)

    per_feature = []
    for w in windows:
        x = torch.from_numpy(w).float()
        x = x.index_select(-1, keep_t)
        x = (x - mean) / std
        x = x.unsqueeze(0).to(device)

        preds = model(x[:, :-1, :])
        targets = x[:, 1:, :]
        err = (preds - targets).abs().squeeze(0)

        pos = [keep.index(i) for i in SHARED]
        err = err.index_select(-1, torch.tensor(pos, dtype=torch.long, device=err.device))
        per_feature.append(err.mean(dim=0).cpu().numpy())

    return np.array(per_feature)


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        base_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "pampos_ablated.yaml") as f:
        abl_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = base_cfg["training"]["seq_len"]

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / base_cfg["data"]["raw_dir"], seq_len=seq_len)
    gen = torch.Generator().manual_seed(base_cfg["training"]["seed"])
    val_size = max(1, int(len(ds) * base_cfg["training"]["val_fraction"]))
    _, val_raw = random_split(ds, [len(ds) - val_size, val_size], generator=gen)
    val_windows = [ds.sequences[i] for i in val_raw.indices]

    print(f"[setup] device {device}")
    print(f"[setup] evaluating on {len(val_windows)} held-out validation windows")
    print(f"[setup] both models are scored ONLY on the five features they share:")
    print(f"[setup]   {SHARED_NAMES}\n")

    ck = REPO_ROOT / base_cfg["paths"]["checkpoint_dir"]
    base_stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    abl_stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats_ablated.npz")

    base_model, base_val = load_model(ck / "pampos_baseline_best.pt", 8, base_cfg, device)
    abl_model, abl_val = load_model(ck / "pampos_ablated_best.pt", 5, abl_cfg, device)

    base_err = shared_feature_errors(
        base_model, val_windows, base_stats["mean"], base_stats["std"], list(range(8)), device)
    abl_err = shared_feature_errors(
        abl_model, val_windows, abl_stats["mean"], abl_stats["std"], SHARED, device)

    print("=" * 78)
    print("REPORTED val_loss -- NOT COMPARABLE")
    print("=" * 78)
    print(f"  8-feature model: {base_val:.6f}   (averaged over 8 targets, incl. point_count)")
    print(f"  5-feature model: {abl_val:.6f}   (averaged over 5 targets)")
    print("  The 5-feature model solves an easier problem, so a lower loss here means")
    print("  fewer hard targets rather than better modelling.\n")

    print("=" * 78)
    print("FAIR COMPARISON -- mean absolute error on the five SHARED features")
    print("=" * 78)
    print(f"{'feature':<16}{'8-feature':<16}{'5-feature':<16}{'better':<12}{'margin'}")
    print("-" * 78)

    wins = {"8-feature": 0, "5-feature": 0}
    per_feature = {}
    for i, name in enumerate(SHARED_NAMES):
        b = float(base_err[:, i].mean())
        a = float(abl_err[:, i].mean())
        better = "8-feature" if b < a else "5-feature"
        wins[better] += 1
        margin = abs(b - a) / max(b, a) * 100
        print(f"{name:<16}{b:<16.6f}{a:<16.6f}{better:<12}{margin:.1f}%")
        per_feature[name] = {"eight_feature": b, "five_feature": a, "better": better}

    b_all = float(base_err.mean())
    a_all = float(abl_err.mean())
    print("-" * 78)
    overall = "8-feature" if b_all < a_all else "5-feature"
    print(f"{'OVERALL':<16}{b_all:<16.6f}{a_all:<16.6f}{overall:<12}"
          f"{abs(b_all - a_all) / max(b_all, a_all) * 100:.1f}%")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  per-feature wins: 8-feature {wins['8-feature']}/5, 5-feature {wins['5-feature']}/5")
    print(f"  overall winner  : {overall}")
    print()
    if overall == "8-feature":
        print("  Carrying the sensor-grounding features improves the model's prediction of")
        print("  motion itself. They contribute information rather than merely adding targets.")
    else:
        print("  The model predicts motion BETTER without the sensor-grounding features.")
        print("  On benign data they add noise rather than information. This does not settle")
        print("  whether they help detect attacks -- that needs the attack data -- but it")
        print("  does mean their value cannot be assumed.")
    print()
    print("  Note: this measures benign reconstruction only. Detection performance is a")
    print("  separate question and requires attack-labelled trajectories.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shared_feature_comparison.json", "w") as f:
        json.dump({
            "n_val_windows": len(val_windows),
            "shared_features": SHARED_NAMES,
            "reported_val_loss": {"eight_feature": base_val, "five_feature": abl_val},
            "per_feature": per_feature,
            "overall": {"eight_feature": b_all, "five_feature": a_all, "winner": overall},
            "per_feature_wins": wins,
        }, f, indent=2)
    print(f"\n[done] saved to outputs/results/shared_feature_comparison.json")


if __name__ == "__main__":
    main()