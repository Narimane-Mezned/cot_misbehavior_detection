import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.losses import huber_loss
from src.data_pipeline.deepaccident_loader import (
    DeepAccidentBenignDataset,
    NormalizedSequenceDataset,
    compute_dataset_stats,
)

EPOCHS = 100
SHARED = [0, 1, 2, 3, 4]
SHARED_NAMES = ["x", "y", "vx", "vy", "yaw"]


class FeatureSubset(Dataset):
    def __init__(self, base, keep):
        self.base = base
        self.keep = torch.tensor(keep, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i].index_select(-1, self.keep)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_pass(model, loader, optimizer, delta, device, train):
    model.train(mode=train)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1, :], batch[:, 1:, :]
        with torch.set_grad_enabled(train):
            preds = model(inputs)
            loss = huber_loss(preds, targets, delta=delta)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item()
        n += 1
    return total / n


@torch.no_grad()
def shared_error(model, windows, mean, std, keep, device):
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32)
    keep_t = torch.tensor(keep, dtype=torch.long)
    pos = torch.tensor([keep.index(i) for i in SHARED], dtype=torch.long, device=device)

    errs = []
    for w in windows:
        x = torch.from_numpy(w).float().index_select(-1, keep_t)
        x = ((x - mean) / std).unsqueeze(0).to(device)
        err = (model(x[:, :-1, :]) - x[:, 1:, :]).abs().squeeze(0)
        errs.append(err.index_select(-1, pos).mean(dim=0).cpu().numpy())
    return np.array(errs)


def train_model(name, keep, cfg, train_raw, val_raw, device, seed):
    set_seed(seed)

    train_sub = FeatureSubset(train_raw, keep)
    val_sub = FeatureSubset(val_raw, keep)
    mean, std = compute_dataset_stats(train_sub)

    train_loader = DataLoader(NormalizedSequenceDataset(train_sub, mean, std),
                              batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(NormalizedSequenceDataset(val_sub, mean, std),
                            batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = PAMPOS(
        input_dim=len(keep),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg["training"]["lr_scheduler_factor"],
        patience=cfg["training"]["lr_scheduler_patience"])

    print(f"[{name}] {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(keep)} features, {EPOCHS} epochs, no early stopping")

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        tr = epoch_pass(model, train_loader, optimizer, cfg["training"]["huber_delta"], device, True)
        va = epoch_pass(model, val_loader, optimizer, cfg["training"]["huber_delta"], device, False)
        scheduler.step(va)
        history.append({"epoch": epoch, "train": tr, "val": va})

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"[{name}] epoch {epoch:3d}  train {tr:.5f}  val {va:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"[{name}] best val {best_val:.6f} at epoch {best_epoch}\n")
    return model, mean.numpy(), std.numpy(), best_val, best_epoch, history


def main():
    with open(REPO_ROOT / "configs" / "pampos_baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg["training"]["seed"]
    seq_len = cfg["training"]["seq_len"]

    ds = DeepAccidentBenignDataset(data_root=REPO_ROOT / cfg["data"]["raw_dir"], seq_len=seq_len)
    gen = torch.Generator().manual_seed(seed)
    val_size = max(1, int(len(ds) * cfg["training"]["val_fraction"]))
    train_raw, val_raw = random_split(ds, [len(ds) - val_size, val_size], generator=gen)
    val_windows = [ds.sequences[i] for i in val_raw.indices]

    print(f"[setup] device {device}")
    print(f"[setup] {len(train_raw)} train / {len(val_raw)} val windows")
    print(f"[setup] both models: identical split, identical seed, {EPOCHS} epochs each")
    print(f"[setup] early stopping DISABLED so neither model gets more budget\n")

    m8, mean8, std8, val8, ep8, hist8 = train_model(
        "8-feature", list(range(8)), cfg, train_raw, val_raw, device, seed)
    m5, mean5, std5, val5, ep5, hist5 = train_model(
        "5-feature", SHARED, cfg, train_raw, val_raw, device, seed)

    e8 = shared_error(m8, val_windows, mean8, std8, list(range(8)), device)
    e5 = shared_error(m5, val_windows, mean5, std5, SHARED, device)

    print("=" * 78)
    print("EQUAL BUDGET -- both trained 100 epochs, best checkpoint taken from each")
    print("=" * 78)
    print(f"  8-feature best val {val8:.6f} at epoch {ep8}")
    print(f"  5-feature best val {val5:.6f} at epoch {ep5}")
    print("  (still not directly comparable -- different numbers of targets)\n")

    print("=" * 78)
    print("FAIR COMPARISON -- mean absolute error on the five SHARED features")
    print("=" * 78)
    print(f"{'feature':<14}{'8-feature':<16}{'5-feature':<16}{'better':<12}{'margin'}")
    print("-" * 78)

    wins = {"8-feature": 0, "5-feature": 0}
    per_feature = {}
    for i, n in enumerate(SHARED_NAMES):
        b, a = float(e8[:, i].mean()), float(e5[:, i].mean())
        better = "8-feature" if b < a else "5-feature"
        wins[better] += 1
        print(f"{n:<14}{b:<16.6f}{a:<16.6f}{better:<12}"
              f"{abs(b - a) / max(b, a) * 100:.1f}%")
        per_feature[n] = {"eight": b, "five": a, "better": better}

    b_all, a_all = float(e8.mean()), float(e5.mean())
    overall = "8-feature" if b_all < a_all else "5-feature"
    margin = abs(b_all - a_all) / max(b_all, a_all) * 100
    print("-" * 78)
    print(f"{'OVERALL':<14}{b_all:<16.6f}{a_all:<16.6f}{overall:<12}{margin:.1f}%")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  per-feature wins : 8-feature {wins['8-feature']}/5, 5-feature {wins['5-feature']}/5")
    print(f"  overall winner   : {overall} by {margin:.1f}%")
    print()
    print("  Earlier run, with unequal budgets (8-feature stopped at 64 epochs, 5-feature")
    print("  ran 100): 5-feature won 5/5 by 14.3%. Compare against that to see how much")
    print("  of the original margin was the extra training rather than the feature set.")
    print()
    print("  This measures benign motion reconstruction only. It does not measure")
    print("  detection, which is what the sensor-grounding features were chosen for.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "equal_budget_comparison.json", "w") as f:
        json.dump({
            "epochs": EPOCHS,
            "early_stopping": False,
            "n_val_windows": len(val_windows),
            "best_val_loss": {"eight": val8, "five": val5},
            "best_epoch": {"eight": ep8, "five": ep5},
            "per_feature": per_feature,
            "overall": {"eight": b_all, "five": a_all,
                        "winner": overall, "margin_pct": margin},
            "per_feature_wins": wins,
            "history": {"eight": hist8, "five": hist5},
        }, f, indent=2)
    print(f"\n[done] saved to outputs/results/equal_budget_comparison.json")


if __name__ == "__main__":
    main()