import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.dreaming import dream_rollout
from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score
from src.data_pipeline.deepaccident_loader import DeepAccidentBenignDataset


def compute_horizon_scores(model, windows: list, seq_len: int, max_horizon: int, feature_mae: torch.Tensor, device: str, mean: torch.Tensor, std: torch.Tensor, k: int = 3) -> dict:
    horizon_scores = {h: [] for h in range(1, max_horizon + 1)}

    for window in windows:
        if window.shape[0] < seq_len + max_horizon:
            continue

        x = torch.from_numpy(window.astype(np.float32)).to(device)
        x = (x - mean) / std

        seed_seq = x[:seq_len].unsqueeze(0)
        ground_truth = x[seq_len:seq_len + max_horizon].unsqueeze(0)

        imagined = dream_rollout(model, seed_seq, max_horizon)

        for h in range(1, max_horizon + 1):
            step_pred = imagined[:, h - 1:h, :]
            step_true = ground_truth[:, h - 1:h, :]
            errors = per_feature_errors(step_pred, step_true)
            normalized = normalize_errors(errors, feature_mae)
            score = topk_anomaly_score(normalized, k=k).mean().item()
            horizon_scores[h].append(score)

    return horizon_scores


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = config["training"]["seq_len"]
    max_horizon = 5
    percentile = config["scoring"]["threshold_percentile"]
    k = config["scoring"]["topk"]

    model = PAMPOS(
        input_dim=config["model"]["input_dim"], d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"], num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"], dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint_path = REPO_ROOT / config["paths"]["checkpoint_dir"] / "pampos_baseline_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[setup] Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    stats_path = REPO_ROOT / "data" / "processed" / "feature_stats.npz"
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)

    data_root = REPO_ROOT / config["data"]["raw_dir"]
    full_seq_len = seq_len + max_horizon
    dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=full_seq_len)
    print(f"[setup] Found {len(dataset)} windows of length {full_seq_len} (seed {seq_len} + horizon {max_horizon})")

    windows = [dataset.sequences[i] for i in range(len(dataset))]

    print("[setup] Computing single-step feature MAE from seed windows (for normalization)...")
    seed_windows_normalized = []
    for w in windows[:300]:
        x = torch.from_numpy(w[:seq_len].astype(np.float32)).to(device)
        x = (x - mean) / std
        seed_windows_normalized.append(x)

    errors_list = []
    with torch.no_grad():
        for x in seed_windows_normalized:
            inputs = x[:-1].unsqueeze(0)
            targets = x[1:].unsqueeze(0)
            preds = model(inputs)
            errors_list.append(per_feature_errors(preds, targets))
    feature_mae = torch.cat(errors_list, dim=0).mean(dim=(0, 1))
    print(f"[setup] Feature MAE: {feature_mae.cpu().tolist()}")

    print(f"\n[run] Computing per-horizon anomaly scores on {len(windows)} real windows...")
    horizon_scores = compute_horizon_scores(model, windows, seq_len, max_horizon, feature_mae, device, mean, std, k=k)

    per_horizon_thresholds = {}
    print(f"\n{'Horizon':<10}{'N':<8}{'Mean':<10}{'Std':<10}{'Threshold (p' + str(percentile) + ')':<20}")
    for h in range(1, max_horizon + 1):
        scores = horizon_scores[h]
        if not scores:
            print(f"{h:<10}{'0':<8}-- no eligible windows --")
            continue
        threshold = float(np.percentile(scores, percentile))
        per_horizon_thresholds[h] = threshold
        print(f"{h:<10}{len(scores):<8}{np.mean(scores):<10.4f}{np.std(scores):<10.4f}{threshold:<20.4f}")

    single_step_threshold = per_horizon_thresholds.get(1)
    if single_step_threshold:
        print(f"\n[check] Single-step threshold (h=1): {single_step_threshold:.4f}")
        print("[check] If later horizons' thresholds are reused as this same single-step value,")
        print("[check] that would be the exact mistake this calibration is meant to prevent.")
        for h, t in per_horizon_thresholds.items():
            if h > 1:
                pct_diff = (t - single_step_threshold) / single_step_threshold * 100
                print(f"[check] Horizon {h} threshold differs from single-step by {pct_diff:+.1f}%")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_horizon_thresholds.json"
    with open(output_path, "w") as f:
        json.dump({
            "thresholds": per_horizon_thresholds,
            "percentile": percentile,
            "topk": k,
            "num_windows_used": len(windows),
            "feature_mae": feature_mae.cpu().tolist(),
        }, f, indent=2)

    print(f"\n[done] Saved per-horizon thresholds to {output_path}")


if __name__ == "__main__":
    main()