import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AttributeInferenceBlackBoxAttack:
    def __init__(
        self,
        input_dim: int,
        attribute_index: int,
        attribute_values: Optional[List[float]] = None,
        learning_rate: float = 0.5,
        train_epochs: int = 2000,
        l2_reg: float = 1e-4,
        name: str = "attribute_inference_black_box",
    ):
        self.input_dim = input_dim
        self.attribute_index = attribute_index
        self.attribute_values = list(attribute_values) if attribute_values is not None else [0.0, 1.0]
        self.value_to_class = {v: i for i, v in enumerate(self.attribute_values)}
        self.num_classes = len(self.attribute_values)
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.l2_reg = l2_reg
        self.name = name

        if not (0 <= attribute_index < input_dim):
            raise ValueError(f"attribute_index {attribute_index} out of range for input_dim {input_dim}")

        self.known_dim = input_dim - 1
        output_dim = 1
        self.attack_input_dim = self.known_dim + output_dim

        self.weights = np.zeros((self.attack_input_dim, self.num_classes))
        self.bias = np.zeros(self.num_classes)
        self.x_mean = np.zeros(self.attack_input_dim)
        self.x_std = np.ones(self.attack_input_dim)

        self.fitted = False
        self.train_accuracy: Optional[float] = None
        self.baseline_accuracy: Optional[float] = None
        self.eval_accuracy: Optional[float] = None
        self.eval_baseline_accuracy: Optional[float] = None
        self.start_time = None
        self.end_time = None

    def _split_state(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        known = np.delete(state, self.attribute_index)
        sensitive = float(state[self.attribute_index])
        return known, sensitive

    def _oracle_output(self, target_query_func: Callable[[np.ndarray], object], full_state: np.ndarray) -> np.ndarray:
        raw = target_query_func(full_state)
        return np.atleast_1d(np.asarray(raw, dtype=np.float64))

    def _transform_output(self, output: np.ndarray) -> np.ndarray:
        if output.shape[-1] == 1:
            p = np.clip(output, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        return output

    def _attack_features(self, known: np.ndarray, output: np.ndarray) -> np.ndarray:
        return np.concatenate([known, self._transform_output(output)], axis=-1)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X_std = (X - self.x_mean) / self.x_std
        logits = X_std @ self.weights + self.bias
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _train_classifier(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X_std = (X - self.x_mean) / self.x_std

        n = X.shape[0]
        for _ in range(self.train_epochs):
            preds = self._forward(X)
            error = preds - Y
            grad_w = X_std.T @ error / n + self.l2_reg * self.weights
            grad_b = error.mean(axis=0)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def fit(self, aux_states: np.ndarray, target_query_func: Callable[[np.ndarray], object]) -> "AttributeInferenceBlackBoxAttack":
        self.start_time = datetime.now()
        aux_states = np.asarray(aux_states, dtype=np.float64)

        X_attack, y_attack = [], []
        for state in aux_states:
            known, sensitive = self._split_state(state)
            if sensitive not in self.value_to_class:
                continue
            output = self._oracle_output(target_query_func, state)
            X_attack.append(self._attack_features(known, output))
            y_attack.append(self.value_to_class[sensitive])

        if not X_attack:
            raise ValueError("No auxiliary samples matched attribute_values -- check attribute_index/attribute_values")

        X_attack = np.array(X_attack)
        y_attack = np.array(y_attack)
        Y_onehot = np.zeros((len(y_attack), self.num_classes))
        Y_onehot[np.arange(len(y_attack)), y_attack] = 1.0

        self._train_classifier(X_attack, Y_onehot)
        self.fitted = True

        preds = np.argmax(self._forward(X_attack), axis=1)
        self.train_accuracy = float(np.mean(preds == y_attack))
        counts = np.bincount(y_attack, minlength=self.num_classes)
        self.baseline_accuracy = float(counts.max() / len(y_attack))

        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"AttributeInferenceBlackBox fit complete! train_accuracy={self.train_accuracy:.3f} vs baseline={self.baseline_accuracy:.3f} | Time: {elapsed:.2f}s")
        return self

    def infer(self, known_features: np.ndarray, observed_output: np.ndarray) -> np.ndarray:
        known_features = np.atleast_2d(np.asarray(known_features, dtype=np.float64))
        output = np.atleast_2d(np.asarray(observed_output, dtype=np.float64))
        if output.shape[0] != known_features.shape[0]:
            output = np.repeat(output, known_features.shape[0], axis=0)

        X = self._attack_features(known_features, output)
        class_idx = np.argmax(self._forward(X), axis=1)
        return np.array([self.attribute_values[i] for i in class_idx])

    def evaluate_accuracy(self, eval_states: np.ndarray, target_query_func: Callable[[np.ndarray], object]) -> Dict:
        eval_states = np.asarray(eval_states, dtype=np.float64)
        known_list, true_list, output_list = [], [], []
        for state in eval_states:
            known, sensitive = self._split_state(state)
            if sensitive not in self.value_to_class:
                continue
            output = self._oracle_output(target_query_func, state)
            known_list.append(known)
            true_list.append(sensitive)
            output_list.append(output)

        if not known_list:
            raise ValueError("No eval samples matched attribute_values -- check attribute_index/attribute_values")

        known_arr = np.array(known_list)
        output_arr = np.array(output_list)
        true_arr = np.array(true_list)

        predicted = self.infer(known_arr, output_arr)
        accuracy = float(np.mean(predicted == true_arr))

        true_classes = np.array([self.value_to_class[v] for v in true_arr])
        counts = np.bincount(true_classes, minlength=self.num_classes)
        eval_baseline = float(counts.max() / len(true_classes))

        self.eval_accuracy = accuracy
        self.eval_baseline_accuracy = eval_baseline
        return {
            "accuracy": accuracy,
            "baseline_accuracy": eval_baseline,
            "advantage_over_baseline": accuracy - eval_baseline,
            "num_samples": int(len(true_arr)),
        }

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "attribute_index": self.attribute_index,
            "attribute_values": self.attribute_values,
            "num_classes": self.num_classes,
            "train_epochs": self.train_epochs,
            "train_accuracy": self.train_accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "eval_accuracy": self.eval_accuracy,
            "eval_baseline_accuracy": self.eval_baseline_accuracy,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return {
            **self.get_statistics(),
            "attack_weights": self.weights.tolist(),
            "attack_bias": self.bias.tolist(),
        }

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_attribute_inference_black_box_attack(input_dim: int, attribute_index: int, attribute_values: Optional[List[float]] = None) -> AttributeInferenceBlackBoxAttack:
    return AttributeInferenceBlackBoxAttack(input_dim=input_dim, attribute_index=attribute_index, attribute_values=attribute_values)


def flatten_window_query_func(pampos_target, seq_len: int, n_features: int):
    def query(flat_state: np.ndarray) -> float:
        window = flat_state.reshape(seq_len, n_features)
        proba = pampos_target.predict_proba(window)
        return proba[1]
    return query


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_dim = 5
    true_weights = np.array([-0.5, 0.2, 1.0, -0.03, 0.3])

    def target_query_func(state: np.ndarray) -> float:
        logits = state @ true_weights
        return 1.0 / (1.0 + np.exp(-logits))

    def sample_states(n: int) -> np.ndarray:
        speed = np.random.uniform(0.0, 30.0, n)
        accel = np.random.uniform(-5.0, 5.0, n)
        ttc = np.random.uniform(0.0, 20.0, n)
        road_type = np.random.choice([0.0, 1.0, 2.0, 3.0], size=n)
        weather = np.random.choice([0.0, 1.0, 2.0], size=n)
        return np.stack([speed, accel, ttc, road_type, weather], axis=1)

    aux_states = sample_states(3000)
    eval_states = sample_states(1000)

    attack = AttributeInferenceBlackBoxAttack(input_dim=5, attribute_index=3, attribute_values=[0.0, 1.0, 2.0, 3.0])
    attack.fit(aux_states, target_query_func)
    results = attack.evaluate_accuracy(eval_states, target_query_func)

    print(f"\nTrain accuracy:    {attack.train_accuracy:.3f} (baseline {attack.baseline_accuracy:.3f})")
    print(f"Held-out accuracy: {results['accuracy']:.3f} (baseline {results['baseline_accuracy']:.3f})")
    print(f"Advantage over baseline: {results['advantage_over_baseline']:+.3f}")