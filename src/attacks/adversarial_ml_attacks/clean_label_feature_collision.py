import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.adversarial_ml_attacks.poisoning_common import FEATURE_NAMES, NON_NEGATIVE_FEATURES

logger = logging.getLogger(__name__)


class CleanLabelFeatureCollisionAttack:
    def __init__(
        self,
        epsilon: float = 0.15,
        direction: Optional[Dict[str, float]] = None,
        name: str = "clean_label_feature_collision",
    ):
        self.epsilon = epsilon
        self.direction = direction or {
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "yaw": 0.0,
            "point_count": -1.0,
            "is_camera_visible": -1.0,
            "distance_to_ego": +1.0,
        }
        self.name = name
        self.shift_vector: Optional[Dict[str, float]] = None
        self.start_time = None
        self.end_time = None

    def poison(self, windows: List[np.ndarray]) -> List[np.ndarray]:
        self.start_time = datetime.now()
        if not windows:
            raise ValueError("Cannot poison an empty set of windows.")

        all_points = np.concatenate([w.reshape(-1, w.shape[-1]) for w in windows], axis=0)
        stds = all_points.std(axis=0)
        stds[stds == 0] = 1e-6

        shift = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.shift_vector = {}
        for i, name in enumerate(FEATURE_NAMES):
            sign = self.direction.get(name, 0.0)
            s = sign * self.epsilon * stds[i]
            shift[i] = s
            self.shift_vector[name] = float(s)

        poisoned_windows = []
        for window in windows:
            shifted = window + shift
            for i, name in enumerate(FEATURE_NAMES):
                if name in NON_NEGATIVE_FEATURES:
                    shifted[:, i] = np.maximum(0.0, shifted[:, i])
            poisoned_windows.append(shifted.astype(np.float32))

        self.end_time = datetime.now()
        logger.info(f"Clean Label Feature Collision Attack: shifted all points by {self.shift_vector}")
        return poisoned_windows

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "epsilon": self.epsilon,
            "direction": self.direction,
            "shift_vector": self.shift_vector,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return self.get_statistics()

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_clean_label_feature_collision_attack(epsilon: float = 0.15) -> CleanLabelFeatureCollisionAttack:
    return CleanLabelFeatureCollisionAttack(epsilon=epsilon)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(1)
    windows = [rng.normal(0, 1, size=(10, 8)).astype(np.float32) for _ in range(50)]

    def mahalanobis_all(features, mean, cov_inv):
        z = features - mean
        return np.sqrt(np.einsum("ij,jk,ik->i", z, cov_inv, z))

    attack_point = np.array([0.0, 0.0, 2.0, 0.5, 0.0, 5.0, 0.0, 40.0])

    all_before = np.concatenate([w.reshape(-1, w.shape[-1]) for w in windows], axis=0)
    mean_before = all_before.mean(axis=0)
    cov_inv_before = np.linalg.pinv(np.cov(all_before.T))
    heldout_before = mahalanobis_all(all_before, mean_before, cov_inv_before)
    tau_95_before = np.percentile(heldout_before, 95)
    attack_dist_before = mahalanobis_all(attack_point[None, :], mean_before, cov_inv_before)[0]

    attack = create_clean_label_feature_collision_attack(epsilon=0.15)
    poisoned_windows = attack.poison(windows)
    all_after = np.concatenate([w.reshape(-1, w.shape[-1]) for w in poisoned_windows], axis=0)
    mean_after = all_after.mean(axis=0)
    cov_inv_after = np.linalg.pinv(np.cov(all_after.T))
    heldout_after = mahalanobis_all(all_after, mean_after, cov_inv_after)
    tau_95_after = np.percentile(heldout_after, 95)
    attack_dist_after = mahalanobis_all(attack_point[None, :], mean_after, cov_inv_after)[0]

    print(f"\nShift vector applied to every point: {attack.shift_vector}")
    print(f"\ntau_95 BEFORE poisoning: {tau_95_before:.2f}")
    print(f"tau_95 AFTER poisoning:  {tau_95_after:.2f}")
    print(f"\nAttack point distance BEFORE: {attack_dist_before:.2f} ({'FLAGGED' if attack_dist_before > tau_95_before else 'missed'})")
    print(f"Attack point distance AFTER:  {attack_dist_after:.2f} ({'FLAGGED' if attack_dist_after > tau_95_after else 'missed'})")