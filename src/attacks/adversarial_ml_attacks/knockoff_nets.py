import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class KnockoffNetsAttack:
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        mode: str = "classification",
        query_budget: int = 500,
        strategy: str = "adaptive",
        batch_size: int = 25,
        pool_multiplier: int = 4,
        learning_rate: float = 0.1,
        train_epochs_per_round: int = 200,
        l2_reg: float = 1e-3,
        input_bounds: Optional[np.ndarray] = None,
        sampling_scale: float = 3.0,
        name: str = "knockoffnets",
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.query_budget = query_budget
        self.strategy = strategy
        self.batch_size = batch_size
        self.pool_multiplier = pool_multiplier
        self.learning_rate = learning_rate
        self.train_epochs_per_round = train_epochs_per_round
        self.l2_reg = l2_reg
        self.input_bounds = np.asarray(input_bounds) if input_bounds is not None else None
        self.sampling_scale = sampling_scale
        self.name = name

        self.weights = np.zeros((self.input_dim, self.output_dim))
        self.bias = np.zeros(self.output_dim)
        self.x_mean = np.zeros(self.input_dim)
        self.x_std = np.ones(self.input_dim)

        self.queried_X = []
        self.queried_y = []
        self.query_count = 0
        self.fidelity_history = []
        self.final_fidelity = None
        self.start_time = None
        self.end_time = None

    def _sample_candidates(self, n: int) -> np.ndarray:
        if self.input_bounds is not None:
            low, high = self.input_bounds[:, 0], self.input_bounds[:, 1]
            return np.random.uniform(low, high, size=(n, self.input_dim))
        return np.random.randn(n, self.input_dim) * self.sampling_scale

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X_std = (X - self.x_mean) / self.x_std
        logits = X_std @ self.weights + self.bias
        if self.mode == "regression":
            return logits
        if self.output_dim == 1:
            return 1.0 / (1.0 + np.exp(-logits))
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _format_target(self, raw) -> np.ndarray:
        if self.mode == "regression":
            return np.atleast_1d(np.asarray(raw, dtype=np.float64))
        if self.output_dim == 1:
            return np.array([float(raw)])
        raw_arr = np.asarray(raw)
        if raw_arr.ndim == 0:
            onehot = np.zeros(self.output_dim)
            onehot[int(raw_arr)] = 1.0
            return onehot
        return raw_arr.astype(np.float64)

    def _train_surrogate(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X_std = (X - self.x_mean) / self.x_std

        n = X.shape[0]
        for _ in range(self.train_epochs_per_round):
            preds = self._forward(X)
            error = preds - Y
            grad_w = X_std.T @ error / n + self.l2_reg * self.weights
            grad_b = error.mean(axis=0)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def _select_batch_uncertainty(self, pool: np.ndarray, batch_size: int) -> np.ndarray:
        probs = self._forward(pool)
        if self.output_dim == 1:
            margin = np.abs(probs[:, 0] - 0.5)
        else:
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        idx = np.argsort(margin)[:batch_size]
        return pool[idx]

    def _select_batch_diversity(self, pool: np.ndarray, batch_size: int) -> np.ndarray:
        if len(self.queried_X) > 0:
            ref = np.array(self.queried_X)
            min_dist = np.linalg.norm(pool[:, None, :] - ref[None, :, :], axis=2).min(axis=1)
        else:
            min_dist = np.linalg.norm(pool, axis=1)

        remaining_mask = np.ones(len(pool), dtype=bool)
        selected_idx = []
        for _ in range(min(batch_size, len(pool))):
            idx = int(np.argmax(np.where(remaining_mask, min_dist, -np.inf)))
            selected_idx.append(idx)
            remaining_mask[idx] = False
            new_dist = np.linalg.norm(pool - pool[idx], axis=1)
            min_dist = np.minimum(min_dist, new_dist)
        return pool[selected_idx]

    def extract(
        self,
        target_query_func: Callable[[np.ndarray], object],
        x_eval: Optional[np.ndarray] = None,
        target_eval_func: Optional[Callable[[np.ndarray], object]] = None,
    ) -> "KnockoffNetsAttack":
        self.start_time = datetime.now()
        self.query_count = 0
        self.queried_X = []
        self.queried_y = []
        self.fidelity_history = []
        round_num = 0

        while self.query_count < self.query_budget:
            current_batch_size = min(self.batch_size, self.query_budget - self.query_count)
            pool_size = max(current_batch_size * self.pool_multiplier, current_batch_size)
            pool = self._sample_candidates(pool_size)

            if self.strategy == "adaptive" and len(self.queried_X) > 0:
                if self.mode == "classification":
                    batch = self._select_batch_uncertainty(pool, current_batch_size)
                else:
                    batch = self._select_batch_diversity(pool, current_batch_size)
            else:
                batch = pool[:current_batch_size]

            for x in batch:
                raw = target_query_func(x)
                self.queried_X.append(x)
                self.queried_y.append(self._format_target(raw))
                self.query_count += 1

            self._train_surrogate(np.array(self.queried_X), np.array(self.queried_y))
            round_num += 1

            if x_eval is not None and target_eval_func is not None:
                fidelity = self.evaluate_fidelity(x_eval, target_eval_func)
                self.fidelity_history.append({"queries": self.query_count, "fidelity": fidelity})
                logger.info(f"Round {round_num} | queries: {self.query_count}/{self.query_budget} | fidelity: {fidelity:.3f}")

        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"KnockoffNets extraction complete! Total queries: {self.query_count} | Final fidelity: {self.final_fidelity} | Time: {elapsed:.2f}s")
        return self

    def evaluate_fidelity(self, x_eval: np.ndarray, target_query_func: Callable[[np.ndarray], object]) -> float:
        x_eval = np.asarray(x_eval, dtype=np.float64)
        surrogate_preds = self._forward(x_eval)

        if self.mode == "regression":
            target_vals = np.array([np.atleast_1d(target_query_func(x)) for x in x_eval])
            mse = np.mean((surrogate_preds - target_vals) ** 2)
            var = np.var(target_vals) + 1e-12
            fidelity = float(max(0.0, 1.0 - mse / var))
        else:
            target_raw = [target_query_func(x) for x in x_eval]
            if self.output_dim == 1:
                target_labels = np.array([1 if float(r) >= 0.5 else 0 for r in target_raw])
                surrogate_labels = (surrogate_preds[:, 0] >= 0.5).astype(int)
            else:
                target_labels = np.array([int(np.argmax(r)) if np.ndim(r) > 0 else int(r) for r in target_raw])
                surrogate_labels = np.argmax(surrogate_preds, axis=1)
            fidelity = float(np.mean(surrogate_labels == target_labels))

        self.final_fidelity = fidelity
        return fidelity

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return self._forward(x)

    def sample_query_points(self, n: int) -> np.ndarray:
        return self._sample_candidates(n)

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "strategy": self.strategy,
            "query_budget": self.query_budget,
            "total_queries": self.query_count,
            "batch_size": self.batch_size,
            "train_epochs_per_round": self.train_epochs_per_round,
            "fidelity_history": self.fidelity_history,
            "final_fidelity": self.final_fidelity,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return {
            **self.get_statistics(),
            "surrogate_weights": self.weights.tolist(),
            "surrogate_bias": self.bias.tolist(),
        }

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_knockoffnets_attack(input_dim: int, query_budget: int = 500, strategy: str = "adaptive") -> KnockoffNetsAttack:
    return KnockoffNetsAttack(input_dim=input_dim, output_dim=1, mode="classification", query_budget=query_budget, strategy=strategy)


def pampos_target_query_func(pampos_target, seq_len: int, n_features: int):
    def query(flat_window: np.ndarray) -> float:
        window = flat_window.reshape(seq_len, n_features)
        proba = pampos_target.predict_proba(window)
        return proba[1]
    return query


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_dim = 80

    true_weights = np.random.default_rng(0).normal(0, 0.5, size=input_dim)

    def target_query_func(state: np.ndarray) -> float:
        logits = state @ true_weights
        return 1.0 / (1.0 + np.exp(-logits))

    attack = KnockoffNetsAttack(input_dim=input_dim, output_dim=1, mode="classification", query_budget=400, strategy="adaptive")

    x_eval = attack._sample_candidates(200)
    attack.extract(target_query_func, x_eval=x_eval, target_eval_func=target_query_func)

    print(f"\nFinal fidelity (agreement with target): {attack.final_fidelity:.3f}")
    print(f"Total queries used: {attack.query_count} / {attack.query_budget}")