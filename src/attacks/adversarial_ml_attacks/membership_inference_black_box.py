import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def run_membership_inference_v2(
    pampos_target,
    member_windows: List[np.ndarray],
    nonmember_windows: List[np.ndarray],
) -> Dict:
    rng = np.random.default_rng(42)

    member_scores = np.array([pampos_target.raw_score(w) for w in member_windows])
    nonmember_scores = np.array([pampos_target.raw_score(w) for w in nonmember_windows])

    n_member_calib = len(member_scores) // 2
    n_nonmember_calib = len(nonmember_scores) // 2

    member_idx = rng.permutation(len(member_scores))
    nonmember_idx = rng.permutation(len(nonmember_scores))

    calib_member = member_scores[member_idx[:n_member_calib]]
    eval_member = member_scores[member_idx[n_member_calib:]]
    calib_nonmember = nonmember_scores[nonmember_idx[:n_nonmember_calib]]
    eval_nonmember = nonmember_scores[nonmember_idx[n_nonmember_calib:]]

    print(f"  [direct test] Calibration: member score mean={calib_member.mean():.4f} std={calib_member.std():.4f} "
          f"(n={len(calib_member)}) | nonmember score mean={calib_nonmember.mean():.4f} std={calib_nonmember.std():.4f} "
          f"(n={len(calib_nonmember)})")

    calib_scores = np.concatenate([calib_member, calib_nonmember])
    calib_labels = np.concatenate([np.ones(len(calib_member)), np.zeros(len(calib_nonmember))])

    candidates = np.percentile(calib_scores, np.arange(1, 100))
    best_threshold, best_acc, best_direction = None, 0.0, "below"
    for t in candidates:
        for direction in ["below", "above"]:
            preds = (calib_scores < t).astype(int) if direction == "below" else (calib_scores > t).astype(int)
            acc = np.mean(preds == calib_labels)
            if acc > best_acc:
                best_acc, best_threshold, best_direction = acc, t, direction

    print(f"  [direct test] Calibrated rule: predict MEMBER if score {'<' if best_direction == 'below' else '>'} "
          f"{best_threshold:.4f} (calibration accuracy: {best_acc:.3f})")

    eval_scores = np.concatenate([eval_member, eval_nonmember])
    eval_labels = np.concatenate([np.ones(len(eval_member)), np.zeros(len(eval_nonmember))])
    eval_preds = (eval_scores < best_threshold).astype(int) if best_direction == "below" else (eval_scores > best_threshold).astype(int)

    accuracy = float(np.mean(eval_preds == eval_labels))
    member_mask = eval_labels == 1
    nonmember_mask = eval_labels == 0
    member_recall = float(np.mean(eval_preds[member_mask] == 1)) if member_mask.sum() > 0 else 0.0
    nonmember_recall = float(np.mean(eval_preds[nonmember_mask] == 0)) if nonmember_mask.sum() > 0 else 0.0
    balanced_accuracy = (member_recall + nonmember_recall) / 2.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "baseline_accuracy": 0.5,
        "advantage_over_baseline": balanced_accuracy - 0.5,
        "member_recall": member_recall,
        "nonmember_recall": nonmember_recall,
        "learned_threshold": float(best_threshold),
        "threshold_direction": best_direction,
        "calibration_accuracy": float(best_acc),
        "num_members_eval": int(member_mask.sum()),
        "num_nonmembers_eval": int(nonmember_mask.sum()),
    }


class ShadowSurrogateTargetModel:
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 16,
        learning_rate: float = 0.3,
        train_epochs: int = 1500,
        l2_reg: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.l2_reg = l2_reg

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.5, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, 0.5, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.x_mean = np.zeros(input_dim)
        self.x_std = np.ones(input_dim)
        self.fitted = False

    def _forward(self, X_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z1 = X_std @ self.W1 + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z2))
        return h, p.ravel()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShadowSurrogateTargetModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X_std = (X - self.x_mean) / self.x_std
        n = X.shape[0]

        for _ in range(self.train_epochs):
            z1 = X_std @ self.W1 + self.b1
            h = np.tanh(z1)
            z2 = h @ self.W2 + self.b2
            p = 1.0 / (1.0 + np.exp(-z2))

            dz2 = (p - y) / n
            dW2 = h.T @ dz2 + self.l2_reg * self.W2
            db2 = dz2.sum(axis=0)
            dh = dz2 @ self.W2.T
            dz1 = dh * (1.0 - h ** 2)
            dW1 = X_std.T @ dz1 + self.l2_reg * self.W1
            db1 = dz1.sum(axis=0)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X_std = (X - self.x_mean) / self.x_std
        _, p = self._forward(X_std)
        return p


def sample_synthetic_windows(n: int, seq_len: int, feature_mean: np.ndarray, feature_std: np.ndarray, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    windows = rng.normal(feature_mean, feature_std, size=(n, seq_len, len(feature_mean)))
    flat = windows.reshape(n, -1)

    point_count_idx = 5
    is_visible_idx = 6
    logits = 0.05 * windows[:, :, point_count_idx].mean(axis=1) + 2.0 * windows[:, :, is_visible_idx].mean(axis=1)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (p > 0.5).astype(np.float64)

    return flat, y


class MembershipInferenceBlackBoxAttack:
    def __init__(
        self,
        num_shadow_models: int = 12,
        shadow_train_size: int = 60,
        shadow_test_size: int = 200,
        target_factory: Optional[Callable[[], ShadowSurrogateTargetModel]] = None,
        attack_learning_rate: float = 0.5,
        attack_train_epochs: int = 1500,
        attack_l2_reg: float = 1e-4,
        name: str = "membership_inference_black_box",
    ):
        self.num_shadow_models = num_shadow_models
        self.shadow_train_size = shadow_train_size
        self.shadow_test_size = shadow_test_size
        self.target_factory = target_factory or (lambda: ShadowSurrogateTargetModel())
        self.attack_learning_rate = attack_learning_rate
        self.attack_train_epochs = attack_train_epochs
        self.attack_l2_reg = attack_l2_reg
        self.name = name

        self.attack_input_dim = 1
        self.num_classes = 2
        self.weights = np.zeros((self.attack_input_dim, self.num_classes))
        self.bias = np.zeros(self.num_classes)
        self.x_mean = np.zeros(self.attack_input_dim)
        self.x_std = np.ones(self.attack_input_dim)

        self.fitted = False
        self.train_accuracy: Optional[float] = None
        self.shadow_member_count: int = 0
        self.shadow_nonmember_count: int = 0
        self.eval_accuracy: Optional[float] = None
        self.eval_balanced_accuracy: Optional[float] = None
        self.eval_baseline_accuracy: Optional[float] = None
        self.eval_member_recall: Optional[float] = None
        self.eval_nonmember_recall: Optional[float] = None
        self.start_time = None
        self.end_time = None

    def _attack_features(self, true_label: np.ndarray, output_p: np.ndarray) -> np.ndarray:
        true_label = np.asarray(true_label, dtype=np.float64)
        output_p = np.asarray(output_p, dtype=np.float64)
        confidence_in_label = np.where(true_label == 1.0, output_p, 1.0 - output_p)
        return confidence_in_label.reshape(-1, 1)

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
        for _ in range(self.attack_train_epochs):
            preds = self._forward(X)
            error = preds - Y
            grad_w = X_std.T @ error / n + self.attack_l2_reg * self.weights
            grad_b = error.mean(axis=0)
            self.weights -= self.attack_learning_rate * grad_w
            self.bias -= self.attack_learning_rate * grad_b

    def fit(self, sample_data_func: Callable[[int], Tuple[np.ndarray, np.ndarray]]) -> "MembershipInferenceBlackBoxAttack":
        self.start_time = datetime.now()

        X_attack: List[np.ndarray] = []
        y_attack: List[np.ndarray] = []

        for _ in range(self.num_shadow_models):
            X_train, y_train = sample_data_func(self.shadow_train_size)
            X_test, y_test = sample_data_func(self.shadow_test_size)

            shadow = self.target_factory()
            shadow.fit(X_train, y_train)

            p_train = shadow.predict_proba(X_train)
            X_attack.append(self._attack_features(y_train, p_train))
            y_attack.append(np.ones(len(y_train)))

            p_test = shadow.predict_proba(X_test)
            X_attack.append(self._attack_features(y_test, p_test))
            y_attack.append(np.zeros(len(y_test)))

        X_attack_arr = np.concatenate(X_attack, axis=0)
        y_attack_arr = np.concatenate(y_attack, axis=0).astype(int)
        self.shadow_member_count = int(np.sum(y_attack_arr == 1))
        self.shadow_nonmember_count = int(np.sum(y_attack_arr == 0))

        minority_count = min(self.shadow_member_count, self.shadow_nonmember_count)
        member_idx = np.flatnonzero(y_attack_arr == 1)
        nonmember_idx = np.flatnonzero(y_attack_arr == 0)
        rng = np.random.default_rng()
        balanced_idx = np.concatenate([
            rng.choice(member_idx, size=minority_count, replace=False),
            rng.choice(nonmember_idx, size=minority_count, replace=False),
        ])
        X_balanced = X_attack_arr[balanced_idx]
        y_balanced = y_attack_arr[balanced_idx]

        Y_onehot = np.zeros((len(y_balanced), self.num_classes))
        Y_onehot[np.arange(len(y_balanced)), y_balanced] = 1.0

        self._train_classifier(X_balanced, Y_onehot)
        self.fitted = True

        preds = np.argmax(self._forward(X_balanced), axis=1)
        self.train_accuracy = float(np.mean(preds == y_balanced))

        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"MembershipInferenceBlackBox fit complete! shadow_models={self.num_shadow_models} train_accuracy={self.train_accuracy:.3f} | Time: {elapsed:.2f}s")
        return self

    def infer_membership(self, true_label: np.ndarray, target_output_p: np.ndarray) -> np.ndarray:
        X = self._attack_features(true_label, target_output_p)
        X = np.atleast_2d(X)
        return np.argmax(self._forward(X), axis=1)

    def evaluate_against_target(
        self,
        pampos_target,
        member_windows: List[np.ndarray],
        nonmember_windows: List[np.ndarray],
    ) -> Dict:
        p_members = np.array([pampos_target.predict_proba(w)[0] for w in member_windows])
        p_nonmembers = np.array([pampos_target.predict_proba(w)[0] for w in nonmember_windows])

        y_members = np.ones(len(member_windows))
        y_nonmembers = np.ones(len(nonmember_windows))

        pred_members = self.infer_membership(y_members, p_members)
        pred_nonmembers = self.infer_membership(y_nonmembers, p_nonmembers)

        correct = int(np.sum(pred_members == 1)) + int(np.sum(pred_nonmembers == 0))
        total = len(y_members) + len(y_nonmembers)
        accuracy = correct / total if total else 0.0

        member_recall = float(np.mean(pred_members == 1)) if len(y_members) else 0.0
        nonmember_recall = float(np.mean(pred_nonmembers == 0)) if len(y_nonmembers) else 0.0
        balanced_accuracy = (member_recall + nonmember_recall) / 2.0

        self.eval_accuracy = accuracy
        self.eval_balanced_accuracy = balanced_accuracy
        self.eval_baseline_accuracy = 0.5
        self.eval_member_recall = member_recall
        self.eval_nonmember_recall = nonmember_recall

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "baseline_accuracy": 0.5,
            "advantage_over_baseline": balanced_accuracy - 0.5,
            "member_recall": member_recall,
            "nonmember_recall": nonmember_recall,
            "num_members": int(len(member_windows)),
            "num_nonmembers": int(len(nonmember_windows)),
        }

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "num_shadow_models": self.num_shadow_models,
            "shadow_train_size": self.shadow_train_size,
            "shadow_test_size": self.shadow_test_size,
            "shadow_member_count": self.shadow_member_count,
            "shadow_nonmember_count": self.shadow_nonmember_count,
            "attack_train_epochs": self.attack_train_epochs,
            "train_accuracy": self.train_accuracy,
            "eval_accuracy": self.eval_accuracy,
            "eval_balanced_accuracy": self.eval_balanced_accuracy,
            "eval_baseline_accuracy": self.eval_baseline_accuracy,
            "eval_member_recall": self.eval_member_recall,
            "eval_nonmember_recall": self.eval_nonmember_recall,
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


def create_membership_inference_black_box_attack(
    num_shadow_models: int = 12,
    shadow_train_size: int = 60,
    shadow_test_size: int = 200,
) -> MembershipInferenceBlackBoxAttack:
    return MembershipInferenceBlackBoxAttack(
        num_shadow_models=num_shadow_models,
        shadow_train_size=shadow_train_size,
        shadow_test_size=shadow_test_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    seq_len = 10
    n_features = 8
    input_dim = seq_len * n_features
    feature_mean = np.zeros(n_features)
    feature_std = np.ones(n_features)

    def sample_data_func(n):
        flat, y = sample_synthetic_windows(n, seq_len, feature_mean, feature_std)
        return flat, y

    attack = MembershipInferenceBlackBoxAttack(
        target_factory=lambda: ShadowSurrogateTargetModel(input_dim=input_dim)
    )
    attack.fit(sample_data_func)

    print(f"\nShadow-model train accuracy (balanced): {attack.train_accuracy:.3f}")
    print(f"Statistics: {json.dumps(attack.get_statistics(), indent=2)}")