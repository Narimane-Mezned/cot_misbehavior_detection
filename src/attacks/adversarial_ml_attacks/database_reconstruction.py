import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_gradient_blackbox(query_func: Callable[[np.ndarray], float], x: np.ndarray, delta: float = 0.001, batch_size: int = 5) -> Tuple[np.ndarray, int]:
    grad = np.zeros_like(x)
    queries = 0
    for _ in range(batch_size):
        u = np.random.randn(*x.shape)
        u = u / np.linalg.norm(u)
        loss_pos = query_func(x + delta * u)
        loss_neg = query_func(x - delta * u)
        grad += (loss_pos - loss_neg) * u / (2 * delta)
        queries += 2
    return grad / batch_size, queries


class DatabaseTargetModel:
    def __init__(self, input_dim: int = 8, learning_rate: float = 0.5, train_epochs: int = 150, l2_reg: float = 1e-3, seed: Optional[int] = None):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.l2_reg = l2_reg

        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.1, size=input_dim)
        self.bias = 0.0
        self.x_mean = np.zeros(input_dim)
        self.x_std = np.ones(input_dim)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DatabaseTargetModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X_std = (X - self.x_mean) / self.x_std
        n = X.shape[0]

        for _ in range(self.train_epochs):
            logits = X_std @ self.weights + self.bias
            p = 1.0 / (1.0 + np.exp(-logits))
            error = (p - y) / n
            grad_w = X_std.T @ error + self.l2_reg * self.weights
            grad_b = error.sum()
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X_std = (X - self.x_mean) / self.x_std
        logits = X_std @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))


class DatabaseReconstructionAttack:
    def __init__(
        self,
        target_factory: Optional[Callable[[], DatabaseTargetModel]] = None,
        input_bounds: Optional[np.ndarray] = None,
        max_iterations: int = 80,
        learning_rate: float = 0.4,
        l2_reg: float = 0.05,
        finite_diff_delta: float = 0.05,
        finite_diff_batch: int = 6,
        window_length: int = 15,
        threshold: float = 1e-4,
        candidate_labels: Sequence[float] = (0.0, 1.0),
        name: str = "database_reconstruction",
    ):
        self.target_factory = target_factory or (lambda: DatabaseTargetModel())
        self.input_bounds = (
            np.asarray(input_bounds, dtype=np.float64)
            if input_bounds is not None
            else np.array([
                [-100.0, 100.0], [-100.0, 100.0], [-20.0, 20.0], [-20.0, 20.0],
                [-3.14, 3.14], [0.0, 3000.0], [0.0, 1.0], [0.0, 200.0],
            ])
        )
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.finite_diff_delta = finite_diff_delta
        self.finite_diff_batch = finite_diff_batch
        self.window_length = window_length
        self.threshold = threshold
        self.candidate_labels = list(candidate_labels)
        self.name = name

        self.gradient_queries = 0
        self.label_results: Dict[float, Dict] = {}
        self.best_result: Optional[Dict] = None
        self.start_time = None
        self.end_time = None

    def _reconstruct_for_label(self, known_X, known_y, target_output_known, candidate_label, anchor, initial_state) -> Dict:
        def neg_distance(x: np.ndarray) -> float:
            candidate_model = self.target_factory()
            X_aug = np.vstack([known_X, x.reshape(1, -1)])
            y_aug = np.concatenate([known_y, [candidate_label]])
            candidate_model.fit(X_aug, y_aug)
            retrained_output = candidate_model.predict_proba(known_X)
            distance = float(np.sum((retrained_output - target_output_known) ** 2))
            return -distance

        x = np.array(initial_state, dtype=np.float64) if initial_state is not None else anchor.copy()
        loss_history: List[float] = []
        best_x, best_neg_dist = x.copy(), neg_distance(x)
        iterations_since_improvement = 0

        for iteration in range(self.max_iterations):
            grad, n_queries = compute_gradient_blackbox(neg_distance, x, delta=self.finite_diff_delta, batch_size=self.finite_diff_batch)
            self.gradient_queries += n_queries

            grad_norm = np.linalg.norm(grad)
            direction = grad / grad_norm if grad_norm > 1e-12 else grad
            x = x + self.learning_rate * direction
            x = x - self.l2_reg * (x - anchor)
            x = np.clip(x, self.input_bounds[:, 0], self.input_bounds[:, 1])

            neg_dist = neg_distance(x)
            distance = -neg_dist
            loss_history.append(distance)

            if neg_dist > best_neg_dist + self.threshold:
                best_neg_dist, best_x = neg_dist, x.copy()
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1
                if iterations_since_improvement >= self.window_length:
                    break

        return {
            "candidate_label": candidate_label,
            "reconstructed_features": best_x.tolist(),
            "achieved_distance": float(-best_neg_dist),
            "iterations_run": len(loss_history),
            "loss_history": loss_history,
        }

    def reconstruct(self, target: DatabaseTargetModel, known_X: np.ndarray, known_y: np.ndarray, initial_state: Optional[np.ndarray] = None) -> Dict:
        self.start_time = datetime.now()

        known_X = np.asarray(known_X, dtype=np.float64)
        known_y = np.asarray(known_y, dtype=np.float64)
        target_output_known = target.predict_proba(known_X)
        anchor = known_X.mean(axis=0)

        self.label_results = {}
        for candidate_label in self.candidate_labels:
            result = self._reconstruct_for_label(known_X, known_y, target_output_known, candidate_label, anchor, initial_state)
            self.label_results[candidate_label] = result

        self.best_result = min(self.label_results.values(), key=lambda r: r["achieved_distance"])
        self.end_time = datetime.now()

        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"DatabaseReconstruction complete! winning_label={self.best_result['candidate_label']} achieved_distance={self.best_result['achieved_distance']:.6f} retrain_queries={self.gradient_queries} | Time: {elapsed:.2f}s")
        return self.best_result

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "max_iterations": self.max_iterations,
            "candidate_labels": self.candidate_labels,
            "retrain_queries": self.gradient_queries,
            "best_result": self.best_result,
            "label_results": self.label_results,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return {**self.get_statistics(), "input_bounds": self.input_bounds.tolist()}

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_database_reconstruction_attack(max_iterations: int = 80, learning_rate: float = 0.4) -> DatabaseReconstructionAttack:
    return DatabaseReconstructionAttack(max_iterations=max_iterations, learning_rate=learning_rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def sample_feature_data(n: int, label_noise: float = 0.1, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-50, 50, n)
        y = rng.uniform(-50, 50, n)
        vx = rng.uniform(-10, 10, n)
        vy = rng.uniform(-10, 10, n)
        yaw = rng.uniform(-3.14, 3.14, n)
        point_count = rng.uniform(0, 1500, n)
        is_visible = rng.choice([0.0, 1.0], size=n)
        distance_to_ego = rng.uniform(0, 100, n)
        X = np.stack([x, y, vx, vy, yaw, point_count, is_visible, distance_to_ego], axis=1)

        logits = 0.01 * point_count + 2.0 * is_visible - 0.02 * distance_to_ego
        p = 1.0 / (1.0 + np.exp(-logits))
        labels = (p > 0.5).astype(np.float64)
        flip = rng.random(n) < label_noise
        labels[flip] = 1.0 - labels[flip]
        return X, labels

    X_full, y_full = sample_feature_data(40, seed=7)
    missing_idx = 0
    missing_row, missing_label = X_full[missing_idx].copy(), y_full[missing_idx]
    known_X = np.delete(X_full, missing_idx, axis=0)
    known_y = np.delete(y_full, missing_idx, axis=0)

    target = DatabaseTargetModel(input_dim=8)
    target.fit(X_full, y_full)

    attack = DatabaseReconstructionAttack(target_factory=lambda: DatabaseTargetModel(input_dim=8))
    result = attack.reconstruct(target, known_X, known_y)

    feature_names = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]
    reconstructed = dict(zip(feature_names, result["reconstructed_features"]))
    true_row = dict(zip(feature_names, missing_row.tolist()))

    print(f"\nTrue missing row (label={missing_label:.0f}):")
    for k, v in true_row.items():
        print(f"   {k}: {v:.2f}")

    print(f"\nReconstructed row (label={result['candidate_label']:.0f}, distance={result['achieved_distance']:.6f}, iterations={result['iterations_run']}):")
    for k, v in reconstructed.items():
        print(f"   {k}: {v:.2f}")

    feature_error = float(np.linalg.norm(np.array(result["reconstructed_features"]) - missing_row))
    print(f"\nFeature L2 error vs. true missing row: {feature_error:.3f}")
    print(f"Label recovered correctly: {result['candidate_label'] == missing_label}")