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


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint...")
    target = load_pampos_target(REPO_ROOT, model_config)

    print("[setup] Building the canonical real train/val split (same seed as train_baseline.py)...")
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)

    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_windows = [full_dataset.sequences[i] for i in train_subset.indices]

    print(f"[setup] Total windows: {len(full_dataset)}, train: {len(train_windows)}")

    calibration_size = min(300, len(train_windows))
    print(f"[setup] Calibrating on {calibration_size} real training windows...")
    target.calibrate(train_windows[:calibration_size])

    print(f"[done] Canonical threshold: {target.threshold:.4f}")
    print(f"[done] Feature MAE: {target.feature_mae.cpu().tolist()}")

    output_dir = REPO_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration.json"

    with open(output_path, "w") as f:
        json.dump({
            "threshold": target.threshold,
            "feature_mae": target.feature_mae.cpu().tolist(),
            "calibration_size": calibration_size,
            "total_windows": len(full_dataset),
            "train_windows": len(train_windows),
            "seed": config["training"]["seed"],
            "note": "Canonical calibration. All scripts should load this file rather than recalibrating independently, to ensure one consistent threshold is used throughout the project.",
        }, f, indent=2)

    print(f"\n[done] Saved canonical calibration to {output_path}")
    print("[done] Update other scripts to load this file instead of recalibrating independently.")


if __name__ == "__main__":
    main()