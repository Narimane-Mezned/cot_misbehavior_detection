import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HopSkipJumpAttack:
    def __init__(
        self,
        max_iterations: int = 20,
        initial_num_evals: int = 20,
        max_num_evals: int = 500,
        gamma: float = 1.0,
        boundary_tolerance: float = 1e-4,
        name: str = "hopskipjump",
    ):
        self.max_iterations = max_iterations
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.gamma = gamma
        self.boundary_tolerance = boundary_tolerance
        self.name = name

        self.perturbation = None
        self.perturbation_history = []
        self.distance_history = []
        self.query_count = 0
        self.start_time = None
        self.end_time = None

    def _decision(self, decision_func: Callable[[np.ndarray], int], x: np.ndarray) -> int:
        self.query_count += 1
        return int(decision_func(x))

    def _binary_search(self, x_benign: np.ndarray, x_adversarial: np.ndarray, decision_func: Callable[[np.ndarray], int]) -> np.ndarray:
        low, high = 0.0, 1.0
        while (high - low) > self.boundary_tolerance:
            mid = (low + high) / 2.0
            x_mid = x_benign + mid * (x_adversarial - x_benign)
            if self._decision(decision_func, x_mid) == 1:
                high = mid
            else:
                low = mid
        return x_benign + high * (x_adversarial - x_benign)

    def _estimate_gradient_direction(self, x_boundary: np.ndarray, decision_func: Callable[[np.ndarray], int], num_evals: int, delta: float) -> np.ndarray:
        dim = x_boundary.shape[0]
        directions = np.random.randn(num_evals, dim)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

        responses = np.zeros(num_evals)
        for i in range(num_evals):
            probe = x_boundary + delta * directions[i]
            responses[i] = 2 * self._decision(decision_func, probe) - 1

        mean_response = responses.mean()
        if abs(mean_response) == 1.0:
            grad = (responses[:, None] * directions).mean(axis=0)
        else:
            grad = ((responses - mean_response)[:, None] * directions).mean(axis=0)

        norm = np.linalg.norm(grad)
        return grad / norm if norm > 1e-12 else directions[0]

    def _geometric_progression_step(self, x_boundary: np.ndarray, direction: np.ndarray, decision_func: Callable[[np.ndarray], int], initial_step: float) -> np.ndarray:
        step = initial_step
        x_candidate = x_boundary + step * direction
        while self._decision(decision_func, x_candidate) == 0:
            step /= 2.0
            x_candidate = x_boundary + step * direction
            if step < 1e-8:
                break
        return x_candidate

    def _find_initial_adversarial(self, x_benign: np.ndarray, decision_func: Callable[[np.ndarray], int], max_tries: int = 1000) -> np.ndarray:
        dim = x_benign.shape[0]
        scale = 1.0
        for _ in range(max_tries):
            candidate = x_benign + np.random.randn(dim) * scale
            if self._decision(decision_func, candidate) == 1:
                return candidate
            scale *= 1.05
        raise RuntimeError("Could not find an initial adversarial point -- target model may be degenerate.")

    def attack(self, x_benign: np.ndarray, decision_func: Callable[[np.ndarray], int], x_adversarial_init: Optional[np.ndarray] = None) -> np.ndarray:
        self.start_time = datetime.now()
        self.query_count = 0
        self.perturbation_history = []
        self.distance_history = []
        x_benign = np.asarray(x_benign, dtype=np.float64)

        if self._decision(decision_func, x_benign) == 1:
            raise ValueError("x_benign must be classified 'safe' (decision 0) -- it is the attack's starting point.")

        if x_adversarial_init is None or self._decision(decision_func, x_adversarial_init) == 0:
            x_adv = self._find_initial_adversarial(x_benign, decision_func)
        else:
            x_adv = np.asarray(x_adversarial_init, dtype=np.float64)

        x_boundary = self._binary_search(x_benign, x_adv, decision_func)
        dist = float(np.linalg.norm(x_boundary - x_benign))
        self.distance_history.append(dist)
        self.perturbation_history.append((x_boundary - x_benign).copy())

        for iteration in range(1, self.max_iterations + 1):
            num_evals = int(min(self.initial_num_evals * np.sqrt(iteration), self.max_num_evals))
            delta = max(dist / np.sqrt(x_benign.shape[0]) * 0.1, 1e-6)

            direction = self._estimate_gradient_direction(x_boundary, decision_func, num_evals, delta)

            initial_step = dist / np.sqrt(iteration) * self.gamma
            x_candidate = self._geometric_progression_step(x_boundary, direction, decision_func, initial_step)

            x_boundary = self._binary_search(x_benign, x_candidate, decision_func)
            dist = float(np.linalg.norm(x_boundary - x_benign))

            self.distance_history.append(dist)
            self.perturbation_history.append((x_boundary - x_benign).copy())

            if iteration % max(1, self.max_iterations // 5) == 0:
                logger.info(f"Iteration {iteration}/{self.max_iterations} | ||delta|| = {dist:.4f} | queries so far: {self.query_count}")

        self.perturbation = x_boundary - x_benign
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"HopSkipJump search complete! Final ||delta||: {self.distance_history[-1]:.4f} | Total queries: {self.query_count} | Time: {elapsed:.2f}s")
        return self.perturbation

    def apply_to_vehicle_state(self, vehicle_state: np.ndarray) -> np.ndarray:
        if self.perturbation is None:
            raise RuntimeError("Perturbation not computed yet. Call attack() first.")
        return vehicle_state + self.perturbation

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "perturbation_norm": float(np.linalg.norm(self.perturbation)) if self.perturbation is not None else None,
            "max_iterations": self.max_iterations,
            "initial_num_evals": self.initial_num_evals,
            "max_num_evals": self.max_num_evals,
            "gamma": self.gamma,
            "total_queries": self.query_count,
            "initial_distance": self.distance_history[0] if self.distance_history else None,
            "final_distance": self.distance_history[-1] if self.distance_history else None,
            "distance_history": self.distance_history,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return {
            **self.get_statistics(),
            "perturbation": self.perturbation.tolist() if self.perturbation is not None else None,
        }

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_hopskipjump_attack(max_iterations: int = 20) -> HopSkipJumpAttack:
    return HopSkipJumpAttack(max_iterations=max_iterations)


def make_pampos_decision_func(pampos_target, seq_len: int, n_features: int):
    def decision(flat_state: np.ndarray) -> int:
        window = flat_state.reshape(seq_len, n_features).astype(np.float32)
        return pampos_target.predict_label(window)
    return decision


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    true_weights = np.array([-0.5, 0.2, 1.0, -0.1, 0.3])

    def decision_oracle(state: np.ndarray) -> int:
        confidence = 1.0 / (1.0 + np.exp(-(state @ true_weights)))
        return 0 if confidence >= 0.5 else 1

    benign_state = np.array([5.0, 0.0, 10.0, 0.0, 0.0])
    assert decision_oracle(benign_state) == 0, "starting point must be 'safe' for this demo"

    attack = HopSkipJumpAttack(max_iterations=15)
    delta = attack.attack(benign_state, decision_oracle)

    print(f"\nMinimal adversarial perturbation: {delta}")
    print(f"Perturbation norm: {np.linalg.norm(delta):.4f}")