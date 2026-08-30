import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.dreaming import measure_error_accumulation
from src.data_pipeline.deepaccident_loader import (
    DeepAccidentBenignDataset,
    NormalizedSequenceDataset,
)


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = REPO_ROOT / config["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = PAMPOS(
        input_dim=config["model"]["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[setup] Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    stats_path = REPO_ROOT / "data" / "processed" / "feature_stats.npz"
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)

    seed_len = 10
    max_horizon = 5
    full_seq_len = seed_len + max_horizon

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    raw_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=full_seq_len)
    print(f"[setup] Found {len(raw_dataset)} windows of length {full_seq_len}")

    if len(raw_dataset) == 0:
        print("[error] No windows long enough for seed_len + max_horizon. Reduce max_horizon or check data.")
        return

    normalized_dataset = NormalizedSequenceDataset(raw_dataset, mean.cpu(), std.cpu())

    num_eval_windows = min(50, len(normalized_dataset))
    rng = np.random.default_rng(42)
    eval_indices = rng.choice(len(normalized_dataset), size=num_eval_windows, replace=False)

    all_errors = []
    for idx in eval_indices:
        full_window = normalized_dataset[idx].unsqueeze(0).to(device)
        seed_seq = full_window[:, :seed_len, :]
        ground_truth_continuation = full_window[:, seed_len:, :]

        per_step_error = measure_error_accumulation(model, seed_seq, ground_truth_continuation, max_horizon)
        all_errors.append(per_step_error.cpu().numpy())

    all_errors = np.stack(all_errors, axis=0)
    mean_per_step = all_errors.mean(axis=0)
    std_per_step = all_errors.std(axis=0)

    print(f"\n[results] Error accumulation over {max_horizon}-step dreaming horizon")
    print(f"[results] Evaluated on {num_eval_windows} real held-out windows\n")
    for step in range(max_horizon):
        print(f"  step {step + 1}: mean abs error (normalized) = {mean_per_step[step]:.4f} (+/- {std_per_step[step]:.4f})")

    results_dir = REPO_ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "dreaming_error_accumulation.npz"
    np.savez(results_path, mean_per_step=mean_per_step, std_per_step=std_per_step, all_errors=all_errors)
    print(f"\n[done] Saved results to {results_path}")


if __name__ == "__main__":
    main()