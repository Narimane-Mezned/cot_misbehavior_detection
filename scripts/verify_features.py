import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data_pipeline"))
from deepaccident_loader import parse_label_file, list_scenarios, get_frame_number, EGO_TRACK_ID


def analyze_scenario_type(root: Path, scenario_type_name: str, max_scenarios: int = 3):
    scenario_type_dir = root / scenario_type_name
    scenario_names = list_scenarios(scenario_type_dir)[:max_scenarios]

    ego_visible_values = []
    other_visible_values = []
    ego_point_counts = []
    other_point_counts = []

    for scenario_name in scenario_names:
        label_dir = scenario_type_dir / "ego_vehicle" / "label" / scenario_name
        frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))

        for frame_file in frame_files:
            parsed = parse_label_file(frame_file)
            for obj in parsed["objects"]:
                if obj["track_id"] == EGO_TRACK_ID:
                    ego_visible_values.append(obj["is_camera_visible"])
                    ego_point_counts.append(obj["point_count"])
                else:
                    other_visible_values.append(obj["is_camera_visible"])
                    other_point_counts.append(obj["point_count"])

    print(f"\n=== {scenario_type_name} ({len(scenario_names)} scenarios) ===")
    print(f"EGO row  -> is_camera_visible values seen: {set(ego_visible_values)}")
    print(f"EGO row  -> point_count range: {min(ego_point_counts)} to {max(ego_point_counts)}, "
          f"mean={np.mean(ego_point_counts):.1f}, std={np.std(ego_point_counts):.1f}")
    print(f"OTHER rows -> is_camera_visible values seen: {set(other_visible_values)} "
          f"(True ratio: {np.mean(other_visible_values):.2%})")
    print(f"OTHER rows -> point_count range: {min(other_point_counts)} to {max(other_point_counts)}, "
          f"mean={np.mean(other_point_counts):.1f}, std={np.std(other_point_counts):.1f}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")

    for scenario_type in ["type1_subtype1_accident", "type1_subtype1_normal"]:
        if (root / scenario_type).exists():
            analyze_scenario_type(root, scenario_type)