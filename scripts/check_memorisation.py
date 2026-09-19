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
from src.model.dreaming import dream_rollout
from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset

SEED_LEN = 10
HORIZON = 3


class Dreamer:
    def __init__(self, ckpt, cfg, mean, std, device):
        self.device = device
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.model = PAMPOS(
            input_dim=cfg["model"]["input_dim"], d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"]).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

    def divergence(self, seed, actual):
        s = ((torch.from_numpy(seed).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        a = ((torch.from_numpy(actual).float() - self.mean) / self.std).unsqueeze(0).to(self.device)
        imagined = dream_rollout(self.model, s, actual.shape[0])
        return float(torch.abs(imagined - a).mean().item())


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = np.load(REPO_ROOT / "data" / "processed" / "feature_stats.npz")
    dreamer = Dreamer(REPO_ROOT / cfg["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt",
                      cfg, stats["mean"], stats["std"], device)

    ds = DeepAccidentBenignDataset(
        data_root=REPO_ROOT / cfg["data"]["raw_dir"],
        seq_len=SEED_LEN + HORIZON)
    gen = torch.Generator().manual_seed(cfg["training"]["seed"])
    val_size = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    train_split, val_split = random_split(ds, [len(ds) - val_size, val_size], generator=gen)

    print(f"[setup] {len(ds)} sequences of length {SEED_LEN + HORIZON}")
    print(f"[setup] {len(train_split)} train / {len(val_split)} val under the training seed")
    print(f"[setup] horizon {HORIZON}, seed length {SEED_LEN}\n")

    def divergences(split, limit=400):
        out = []
        for i in list(split.indices)[:limit]:
            seq = ds.sequences[i]
            if len(seq) < SEED_LEN + HORIZON:
                continue
            out.append(dreamer.divergence(seq[:SEED_LEN], seq[SEED_LEN:SEED_LEN + HORIZON]))
        return np.array(out)

    tr = divergences(train_split)
    va = divergences(val_split)

    print("=" * 78)
    print("1. MEMORISATION CHECK -- rollout divergence on seen vs unseen benign data")
    print("=" * 78)
    print(f"   training sequences (model has seen these) : {tr.mean():.4f} +/- {tr.std():.4f}  (n={len(tr)})")
    print(f"   validation sequences (never seen)         : {va.mean():.4f} +/- {va.std():.4f}  (n={len(va)})")
    ratio = va.mean() / tr.mean() if tr.mean() > 1e-9 else float("inf")
    print(f"   ratio val/train                           : {ratio:.3f}")
    print()
    if ratio > 1.5:
        print("   The model rolls out seen sequences substantially better than unseen ones.")
        print("   Benign divergence in the detection experiment is therefore partly an")
        print("   artifact of those trajectories being training data, and the reported")
        print("   separation is inflated.")
    elif ratio > 1.15:
        print("   Mild memorisation. The detection separation is somewhat optimistic but")
        print("   the effect is modest.")
    else:
        print("   No meaningful memorisation: the model rolls out unseen benign sequences")
        print("   about as well as seen ones, so the detection separation is not explained")
        print("   by the benign trajectories having been in training.")

    print()
    print("=" * 78)
    print("2. SCALE CHECK -- where do the attack divergences sit relative to benign?")
    print("=" * 78)
    p = {q: float(np.percentile(va, q)) for q in (50, 90, 99, 100)}
    print(f"   unseen benign divergence: median {p[50]:.4f}, "
          f"p90 {p[90]:.4f}, p99 {p[99]:.4f}, max {p[100]:.4f}")

    res_path = REPO_ROOT / "outputs" / "results" / "dreaming_detection_8feature.json"
    if res_path.exists():
        print()
        print("   Compare this against the attacked divergences from the detection run.")
        print("   If the attacked scores sit far above the benign maximum, the separation")
        print("   is driven by the attack rather than by ordinary variation.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "memorisation_check.json", "w") as f:
        json.dump({"horizon": HORIZON, "seed_len": SEED_LEN,
                   "train": {"mean": float(tr.mean()), "std": float(tr.std()), "n": len(tr)},
                   "val": {"mean": float(va.mean()), "std": float(va.std()), "n": len(va)},
                   "ratio_val_over_train": float(ratio),
                   "val_percentiles": p}, f, indent=2)
    print(f"\n[done] saved to outputs/results/memorisation_check.json")


if __name__ == "__main__":
    main()