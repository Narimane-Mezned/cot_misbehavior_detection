import numpy as np

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]
NON_NEGATIVE_FEATURES = {"point_count", "is_camera_visible", "distance_to_ego"}


def windows_to_flat_points(windows: list) -> tuple:
    index_map = []
    flat_rows = []
    for window_idx, window in enumerate(windows):
        for timestep_idx in range(window.shape[0]):
            flat_rows.append(window[timestep_idx])
            index_map.append((window_idx, timestep_idx))
    return np.array(flat_rows, dtype=np.float32), index_map


def apply_flat_overrides(windows: list, overrides: dict) -> list:
    poisoned = [w.copy() for w in windows]
    for (window_idx, timestep_idx), new_row in overrides.items():
        poisoned[window_idx][timestep_idx] = new_row
    return poisoned


def trigger_dict_to_vector(trigger_pattern: dict) -> np.ndarray:
    return np.array([trigger_pattern.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)