import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple

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


class MIFaceAttack:
    def __init__(
        self,
        input_dim: int,
        input_bounds: np.ndarray,
        max_iterations: int = 300,
        learning_rate: float = 0.5,
        l2_reg: float = 0.05,
        finite_diff_delta: float = 0.01,
        finite_diff_batch: int = 20,
        window_length: int = 25,
        threshold: float = 1e-4,
        name: str = "miface",
    ):
        self.input_dim = input_dim
        self.input_bounds = np.asarray(input_bounds, dtype=np.float64)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.finite_diff_delta = finite_diff_delta
        self.finite_diff_batch = finite_diff_batch
        self.window_length = window_length
        self.threshold = threshold
        self.name = name

        self.anchor = (self.input_bounds[:, 0] + self.input_bounds[:, 1]) / 2.0
        self.gradient_queries = 0
        self.reconstructions: Dict[int, Dict] = {}
        self.start_time = None
        self.end_time = None

    def invert_class(
        self,
        target_query_func: Callable[[np.ndarray], object],
        target_class: int,
        initial_state: Optional[np.ndarray] = None,
    ) -> Dict:
        def signed_confidence(x: np.ndarray) -> float:
            c = float(target_query_func(x))
            return c if target_class == 1 else (1.0 - c)

        x = np.array(initial_state, dtype=np.float64) if initial_state is not None else self.anchor.copy()
        loss_history: List[float] = []
        best_x, best_conf = x.copy(), signed_confidence(x)
        iterations_since_improvement = 0

        for iteration in range(self.max_iterations):
            grad, n_queries = compute_gradient_blackbox(
                signed_confidence, x, delta=self.finite_diff_delta, batch_size=self.finite_diff_batch
            )
            self.gradient_queries += n_queries

            grad_norm = np.linalg.norm(grad)
            direction = grad / grad_norm if grad_norm > 1e-12 else grad
            x = x + self.learning_rate * direction
            x = x - self.l2_reg * (x - self.anchor)
            x = np.clip(x, self.input_bounds[:, 0], self.input_bounds[:, 1])

            conf = signed_confidence(x)
            loss = 1.0 - conf
            loss_history.append(loss)

            if conf > best_conf + self.threshold:
                best_conf, best_x = conf, x.copy()
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1
                if iterations_since_improvement >= self.window_length:
                    break

        result = {
            "target_class": target_class,
            "reconstructed_state": best_x.tolist(),
            "achieved_confidence": float(best_conf),
            "iterations_run": len(loss_history),
            "loss_history": loss_history,
        }
        self.reconstructions[target_class] = result
        return result

    def invert_all_classes(
        self,
        target_query_func: Callable[[np.ndarray], object],
        classes: Optional[List[int]] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> "MIFaceAttack":
        self.start_time = datetime.now()
        for target_class in (classes or [0, 1]):
            self.invert_class(target_query_func, target_class, initial_state=initial_state)
        self.end_time = datetime.now()

        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"MIFace inversion complete! classes={list(self.reconstructions.keys())} gradient_queries={self.gradient_queries} | Time: {elapsed:.2f}s")
        return self

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "max_iterations": self.max_iterations,
            "gradient_queries": self.gradient_queries,
            "reconstructions": self.reconstructions,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return {
            **self.get_statistics(),
            "input_bounds": self.input_bounds.tolist(),
            "anchor": self.anchor.tolist(),
        }

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_miface_attack(input_dim: int, input_bounds: np.ndarray, max_iterations: int = 300, learning_rate: float = 0.5) -> MIFaceAttack:
    return MIFaceAttack(input_dim=input_dim, input_bounds=input_bounds, max_iterations=max_iterations, learning_rate=learning_rate)


def flatten_window_query_func(pampos_target, seq_len: int, n_features: int):
    def query(flat_state: np.ndarray) -> float:
        window = flat_state.reshape(seq_len, n_features).astype(np.float32)
        proba = pampos_target.predict_proba(window)
        return proba[1]
    return query


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_dim = 5
    true_weights = np.array([-0.5, 0.2, 1.0, -0.1, 0.3])
    input_bounds = np.array([[0.0, 30.0], [-5.0, 5.0], [0.0, 20.0], [0.0, 3.0], [0.0, 3.0]])

    def target_query_func(state: np.ndarray) -> float:
        logits = state @ true_weights
        return 1.0 / (1.0 + np.exp(-logits))

    attack = MIFaceAttack(input_dim=input_dim, input_bounds=input_bounds)
    attack.invert_all_classes(target_query_func)

    feature_names = ["speed", "acceleration", "ttc", "road_type", "weather"]
    for target_class, result in attack.reconstructions.items():
        label = "SAFE" if target_class == 1 else "UNSAFE"
        state = dict(zip(feature_names, result["reconstructed_state"]))
        print(f"\nReconstructed typical {label} state (confidence={result['achieved_confidence']:.3f}, iterations={result['iterations_run']}):")
        for k, v in state.items():
            print(f"   {k}: {v:.2f}")

    print(f"\nTotal gradient queries used: {attack.gradient_queries}")