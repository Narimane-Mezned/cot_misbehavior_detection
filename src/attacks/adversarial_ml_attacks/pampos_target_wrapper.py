import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.pampos import PAMPOS
from src.model.losses import per_feature_errors, normalize_errors, topk_anomaly_score


class PAMPOSTarget:
    def __init__(self, checkpoint_path: Path, feature_stats_path: Path, model_config: dict, device: str = "cpu"):
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = PAMPOS(**model_config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        stats = np.load(feature_stats_path)
        self.norm_mean = torch.from_numpy(stats["mean"]).to(device)
        self.norm_std = torch.from_numpy(stats["std"]).to(device)

        self.feature_mae = None
        self.threshold = None
        self.query_count = 0

    def _normalize(self, window: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(window.astype(np.float32)).to(self.device)
        return (x - self.norm_mean) / self.norm_std

    @torch.no_grad()
    def raw_score(self, window: np.ndarray) -> float:
        self.query_count += 1
        x = self._normalize(window).unsqueeze(0)
        inputs = x[:, :-1, :]
        targets = x[:, 1:, :]
        preds = self.model(inputs)

        errors = per_feature_errors(preds, targets)
        if self.feature_mae is not None:
            errors = normalize_errors(errors, self.feature_mae)
        score = topk_anomaly_score(errors, k=3).mean().item()
        return score

    def calibrate(self, benign_windows: list, k: int = 3, percentile: float = 99.0):
        all_errors = []
        for window in benign_windows:
            x = self._normalize(window).unsqueeze(0)
            inputs = x[:, :-1, :]
            targets = x[:, 1:, :]
            with torch.no_grad():
                preds = self.model(inputs)
            errors = per_feature_errors(preds, targets)
            all_errors.append(errors)

        stacked = torch.cat(all_errors, dim=0)
        self.feature_mae = stacked.mean(dim=(0, 1))

        benign_scores = []
        for window in benign_windows:
            benign_scores.append(self.raw_score(window))
        self.threshold = float(np.percentile(benign_scores, percentile))
        self.query_count = 0

    def predict_proba(self, window: np.ndarray, scale: float = 2.0) -> np.ndarray:
        score = self.raw_score(window)
        if self.threshold is None:
            raise ValueError("Call calibrate() before predict_proba()")
        z = (score - self.threshold) / (scale + 1e-8)
        p_anomalous = 1.0 / (1.0 + np.exp(-z))
        return np.array([1.0 - p_anomalous, p_anomalous])

    def predict_label(self, window: np.ndarray) -> int:
        score = self.raw_score(window)
        if self.threshold is None:
            raise ValueError("Call calibrate() before predict_label()")
        return int(score > self.threshold)


def load_pampos_target(repo_root: Path, model_config: dict, device: str = "cpu") -> PAMPOSTarget:
    checkpoint_path = repo_root / "outputs" / "checkpoints" / "pampos_baseline_best.pt"
    feature_stats_path = repo_root / "data" / "processed" / "feature_stats.npz"
    return PAMPOSTarget(checkpoint_path, feature_stats_path, model_config, device=device)


if __name__ == "__main__":
    model_config = {
        "input_dim": 8, "d_model": 128, "nhead": 8,
        "num_layers": 3, "dim_feedforward": 256, "dropout": 0.1,
    }

    target = load_pampos_target(REPO_ROOT, model_config)

    rng = np.random.default_rng(42)
    benign_windows = [rng.normal(0, 1, size=(10, 8)).astype(np.float32) for _ in range(20)]
    target.calibrate(benign_windows)
    print(f"Threshold: {target.threshold:.4f}")

    test_window = rng.normal(0, 1, size=(10, 8)).astype(np.float32)
    print(f"predict_proba: {target.predict_proba(test_window)}")
    print(f"predict_label: {target.predict_label(test_window)}")
    print(f"Query count: {target.query_count}")