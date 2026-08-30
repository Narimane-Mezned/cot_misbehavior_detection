import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.losses import huber_loss
from src.data_pipeline.deepaccident_loader import (
    DeepAccidentBenignDataset,
    NormalizedSequenceDataset,
    compute_dataset_stats,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def next_step_targets(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch[:, :-1, :]
    targets = batch[:, 1:, :]
    return inputs, targets


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, delta: float, device: str, train: bool) -> float:
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        inputs, targets = next_step_targets(batch)

        with torch.set_grad_enabled(train):
            preds = model(inputs)
            loss = huber_loss(preds, targets, delta=delta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] Using device: {device}")

    model = PAMPOS(
        input_dim=config["model"]["input_dim"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] Model parameters: {n_params:,}")

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_dataset = DeepAccidentBenignDataset(
        data_root=data_root,
        seq_len=config["training"]["seq_len"],
    )
    print(f"[setup] Total benign training windows: {len(full_dataset)}")

    val_fraction = config["training"]["val_fraction"]
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_dataset_raw, val_dataset_raw = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print(f"[setup] Train windows: {len(train_dataset_raw)}, Val windows: {len(val_dataset_raw)}")

    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_path = processed_dir / "feature_stats.npz"

    if stats_path.exists():
        print(f"[setup] Loading existing feature stats from {stats_path}")
        stats = np.load(stats_path)
        mean = torch.from_numpy(stats["mean"])
        std = torch.from_numpy(stats["std"])
    else:
        print("[setup] Computing feature normalization stats from training split...")
        mean, std = compute_dataset_stats(train_dataset_raw)
        np.savez(stats_path, mean=mean.numpy(), std=std.numpy())
        print(f"[setup] Saved feature stats to {stats_path}")

    print(f"[setup] Feature mean: {mean.tolist()}")
    print(f"[setup] Feature std:  {std.tolist()}")

    train_dataset = NormalizedSequenceDataset(train_dataset_raw, mean, std)
    val_dataset = NormalizedSequenceDataset(val_dataset_raw, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["lr_scheduler_factor"],
        patience=config["training"]["lr_scheduler_patience"],
    )

    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "pampos_baseline_best.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, config["training"]["huber_delta"], device, train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, config["training"]["huber_delta"], device, train=False
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"[checkpoint] Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[early stop] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break

    print(f"[done] Best val_loss: {best_val_loss:.4f}")
    print(f"[done] Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()