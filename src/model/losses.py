import torch
import torch.nn as nn


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    loss_fn = nn.HuberLoss(delta=delta)
    return loss_fn(pred, target)


def per_feature_errors(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target)


def normalize_errors(errors: torch.Tensor, feature_mae: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return errors / (feature_mae + eps)


def topk_anomaly_score(normalized_errors: torch.Tensor, k: int = 3) -> torch.Tensor:
    topk_values, _ = torch.topk(normalized_errors, k=k, dim=-1)
    return topk_values.mean(dim=-1)


def compute_feature_mae(errors_benign: torch.Tensor) -> torch.Tensor:
    return errors_benign.mean(dim=tuple(range(errors_benign.dim() - 1)))


def calibrate_threshold(benign_scores: torch.Tensor, percentile: float = 99.0) -> float:
    return torch.quantile(benign_scores.flatten(), percentile / 100.0).item()


def anomaly_pipeline(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_mae: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    errors = per_feature_errors(pred, target)
    normalized = normalize_errors(errors, feature_mae)
    return topk_anomaly_score(normalized, k=k)


if __name__ == "__main__":
    torch.manual_seed(42)

    pred = torch.randn(4, 10, 8)
    target = torch.randn(4, 10, 8)

    loss = huber_loss(pred, target)
    print(f"Huber loss: {loss.item():.4f}")

    benign_pred = torch.randn(100, 10, 8)
    benign_target = benign_pred + torch.randn(100, 10, 8) * 0.1
    benign_errors = per_feature_errors(benign_pred, benign_target)
    feature_mae = compute_feature_mae(benign_errors)
    print(f"Per-feature MAE (benign): {feature_mae}")

    benign_scores = anomaly_pipeline(benign_pred, benign_target, feature_mae, k=3)
    threshold = calibrate_threshold(benign_scores, percentile=99.0)
    print(f"Calibrated threshold (99th pct): {threshold:.4f}")

    attack_pred = torch.randn(4, 10, 8)
    attack_target = attack_pred + torch.randn(4, 10, 8) * 2.0
    attack_scores = anomaly_pipeline(attack_pred, attack_target, feature_mae, k=3)
    print(f"Attack scenario scores: {attack_scores.mean(dim=1)}")
    print(f"Flagged as anomalous: {(attack_scores.mean(dim=1) > threshold)}")