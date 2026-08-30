import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.adversarial_ml_attacks.poisoning_common import (
    FEATURE_NAMES,
    windows_to_flat_points,
    apply_flat_overrides,
    trigger_dict_to_vector,
)

logger = logging.getLogger(__name__)


class BackdoorAttack:
    def __init__(
        self,
        trigger_pattern: Dict[str, float],
        num_poisoned_points: Optional[int] = None,
        fraction_poisoned: float = 0.15,
        name: str = "backdoor",
    ):
        self.trigger_pattern = trigger_pattern
        self.trigger_vector = trigger_dict_to_vector(trigger_pattern)
        self.num_poisoned_points = num_poisoned_points
        self.fraction_poisoned = fraction_poisoned
        self.name = name
        self.poisoned_locations: List[tuple] = []
        self.start_time = None
        self.end_time = None

    def poison(self, windows: List[np.ndarray], seed: Optional[int] = None) -> List[np.ndarray]:
        self.start_time = datetime.now()
        rng = np.random.default_rng(seed)

        flat_points, index_map = windows_to_flat_points(windows)
        n = len(index_map)
        if n == 0:
            raise ValueError("Cannot poison an empty set of windows.")

        target_count = self.num_poisoned_points if self.num_poisoned_points is not None else max(1, int(self.fraction_poisoned * n))
        k = min(target_count, n)
        chosen_flat_indices = sorted(rng.choice(n, size=k, replace=False).tolist())
        self.poisoned_locations = [index_map[i] for i in chosen_flat_indices]

        overrides = {loc: self.trigger_vector.copy() for loc in self.poisoned_locations}
        poisoned_windows = apply_flat_overrides(windows, overrides)

        self.end_time = datetime.now()
        logger.info(f"Backdoor Attack: planted trigger {self.trigger_pattern} at {k}/{n} points")
        return poisoned_windows

    def get_statistics(self) -> Dict:
        elapsed = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "attack_name": self.name,
            "trigger_pattern": self.trigger_pattern,
            "num_poisoned_points": len(self.poisoned_locations),
            "fraction_poisoned": self.fraction_poisoned,
            "elapsed_seconds": elapsed,
        }

    def to_dict(self) -> Dict:
        return self.get_statistics()

    def save_to_file(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_backdoor_attack(
    trigger_pattern: Optional[Dict[str, float]] = None,
    fraction_poisoned: float = 0.15,
) -> BackdoorAttack:
    default_trigger = {
        "x": 0.0,
        "y": 0.0,
        "vx": 2.0,
        "vy": 0.5,
        "yaw": 0.0,
        "point_count": 0.0,
        "is_camera_visible": 0.0,
        "distance_to_ego": 20.0,
    }
    return BackdoorAttack(trigger_pattern=trigger_pattern or default_trigger, fraction_poisoned=fraction_poisoned)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(0)
    windows = [rng.normal(0, 1, size=(10, 8)).astype(np.float32) for _ in range(50)]

    def mahalanobis(x, mean, cov_inv):
        z = x - mean
        return float(np.sqrt(z.dot(cov_inv).dot(z)))

    attack = create_backdoor_attack()
    trigger_vec = attack.trigger_vector

    flat_before, _ = windows_to_flat_points(windows)
    mean_before = flat_before.mean(axis=0)
    cov_inv_before = np.linalg.pinv(np.cov(flat_before.T))
    dist_before = mahalanobis(trigger_vec, mean_before, cov_inv_before)

    poisoned_windows = attack.poison(windows, seed=0)
    flat_after, _ = windows_to_flat_points(poisoned_windows)
    mean_after = flat_after.mean(axis=0)
    cov_inv_after = np.linalg.pinv(np.cov(flat_after.T))
    dist_after = mahalanobis(trigger_vec, mean_after, cov_inv_after)

    print(f"\nTrigger pattern: {attack.trigger_pattern}")
    print(f"Mahalanobis distance of trigger BEFORE poisoning: {dist_before:.2f}")
    print(f"Mahalanobis distance of trigger AFTER poisoning:  {dist_after:.2f}")
    print(f"\nStatistics: {json.dumps(attack.get_statistics(), indent=2)}")