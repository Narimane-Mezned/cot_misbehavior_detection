import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EGO_TRACK_ID = -100
UNTRACKED_TRACK_ID = -1
EXPECTED_LABEL_FIELDS = 13
FRAME_PATTERN = re.compile(r"_(\d+)\.txt$")


def parse_meta(path: Path) -> dict:
    lines = Path(path).read_text().strip().split("\n")

    first_line_tokens = lines[0].split()
    weather = first_line_tokens[0]
    rest_tokens = first_line_tokens[1:]

    def try_parse_token(token: str):
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    parsed_rest = [try_parse_token(t) for t in rest_tokens]

    meta = {
        "weather": weather,
        "collision_header_raw": parsed_rest,
        "designed_collision": False,
    }

    if len(parsed_rest) == 8 and isinstance(parsed_rest[1], str) and isinstance(parsed_rest[3], str):
        meta["designed_collision"] = True
        meta["collision_partner1_id"] = parsed_rest[0]
        meta["collision_partner1_type"] = parsed_rest[1]
        meta["collision_partner2_id"] = parsed_rest[2]
        meta["collision_partner2_type"] = parsed_rest[3]
        meta["collision_metric"] = parsed_rest[4]
        meta["collision_direction_1"] = parsed_rest[5]
        meta["collision_direction_2"] = parsed_rest[6]
        meta["num_frames"] = parsed_rest[7]
    else:
        meta["num_frames"] = parsed_rest[-1] if parsed_rest else None

    for line in lines[1:]:
        line = line.strip()
        if line.startswith("colliding agents:"):
            meta["colliding_agents"] = line.split(":", 1)[1].strip().split()
        elif line.startswith("agents id:"):
            meta["agent_ids"] = [int(x) for x in line.split(":", 1)[1].strip().split()]
        elif line.startswith("road_type:"):
            meta["road_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("another_vehicle_spawn_side:"):
            meta["another_vehicle_spawn_side"] = line.split(":", 1)[1].strip()
        elif line.startswith("ego_vehicle_direction:"):
            meta["ego_vehicle_direction"] = line.split(":", 1)[1].strip()
        elif line.startswith("other_vehicle_direction:"):
            meta["other_vehicle_direction"] = line.split(":", 1)[1].strip()

    return meta


def parse_label_file(path: Path) -> dict:
    lines = Path(path).read_text().strip().split("\n")

    ego_velocity_header = tuple(float(x) for x in lines[0].split())

    objects = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != EXPECTED_LABEL_FIELDS:
            raise ValueError(
                f"Expected {EXPECTED_LABEL_FIELDS} fields, got {len(fields)} in {path}: {line}"
            )

        objects.append({
            "category": fields[0],
            "x": float(fields[1]),
            "y": float(fields[2]),
            "z": float(fields[3]),
            "length": float(fields[4]),
            "width": float(fields[5]),
            "height": float(fields[6]),
            "yaw": float(fields[7]),
            "vx": float(fields[8]),
            "vy": float(fields[9]),
            "track_id": int(fields[10]),
            "point_count": int(fields[11]),
            "is_camera_visible": fields[12] == "True",
        })

    return {"ego_velocity_header": ego_velocity_header, "objects": objects}


def get_frame_number(filename: str) -> int:
    match = FRAME_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"Could not extract frame number from {filename}")
    return int(match.group(1))


def list_scenarios(scenario_type_dir: Path) -> list[str]:
    label_dir = Path(scenario_type_dir) / "ego_vehicle" / "label"
    return sorted(d.name for d in label_dir.iterdir() if d.is_dir())


def agent_feature_vector(obj: dict, ego_obj: dict) -> list[float]:
    distance_to_ego = math.sqrt((obj["x"] - ego_obj["x"]) ** 2 + (obj["y"] - ego_obj["y"]) ** 2)
    return [
        obj["x"],
        obj["y"],
        obj["vx"],
        obj["vy"],
        obj["yaw"],
        float(obj["point_count"]),
        1.0 if obj["is_camera_visible"] else 0.0,
        distance_to_ego,
    ]


def build_agent_sequences(scenario_type_dir: Path, scenario_name: str) -> dict[int, np.ndarray]:
    label_dir = Path(scenario_type_dir) / "ego_vehicle" / "label" / scenario_name
    frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))

    per_track_features = defaultdict(list)

    for frame_file in frame_files:
        parsed = parse_label_file(frame_file)
        ego_obj = next((o for o in parsed["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        if ego_obj is None:
            continue

        for obj in parsed["objects"]:
            if obj["track_id"] == EGO_TRACK_ID:
                continue
            if obj["track_id"] == UNTRACKED_TRACK_ID:
                continue
            per_track_features[obj["track_id"]].append(agent_feature_vector(obj, ego_obj))

    return {
        track_id: np.array(feats, dtype=np.float32)
        for track_id, feats in per_track_features.items()
    }


class DeepAccidentBenignDataset(Dataset):
    def __init__(self, data_root: Path, seq_len: int, min_track_frames: int | None = None):
        self.seq_len = seq_len
        min_track_frames = min_track_frames or seq_len
        self.sequences = []

        data_root = Path(data_root)
        normal_dirs = sorted(data_root.glob("*_normal"))

        for scenario_type_dir in normal_dirs:
            scenario_names = list_scenarios(scenario_type_dir)
            for scenario_name in scenario_names:
                agent_sequences = build_agent_sequences(scenario_type_dir, scenario_name)
                for track_id, full_seq in agent_sequences.items():
                    if len(full_seq) < min_track_frames:
                        continue
                    for start in range(0, len(full_seq) - seq_len + 1, seq_len):
                        self.sequences.append(full_seq[start:start + seq_len])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.sequences[idx])


class NormalizedSequenceDataset(Dataset):
    def __init__(self, base_dataset, mean: torch.Tensor, std: torch.Tensor):
        self.base_dataset = base_dataset
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.base_dataset[idx]
        return (x - self.mean) / self.std


def compute_dataset_stats(dataset, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    all_values = torch.cat([dataset[i] for i in range(len(dataset))], dim=0)
    mean = all_values.mean(dim=0)
    std = all_values.std(dim=0)
    std = torch.clamp(std, min=eps)
    return mean, std


if __name__ == "__main__":
    import sys

    test_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")

    scenario_type_dir = test_root / "type1_subtype1_accident"
    scenario_names = list_scenarios(scenario_type_dir)
    print(f"Found {len(scenario_names)} scenarios in {scenario_type_dir.name}")

    scenario_name = scenario_names[0]
    agent_sequences = build_agent_sequences(scenario_type_dir, scenario_name)

    print(f"\nScenario: {scenario_name}")
    print(f"Tracked non-ego agents: {list(agent_sequences.keys())}")
    for track_id, seq in agent_sequences.items():
        print(f"  track_id={track_id}: {seq.shape[0]} frames, feature shape={seq.shape[1]}")