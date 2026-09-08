import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import (
    parse_label_file,
    parse_meta,
    list_scenarios,
    get_frame_number,
    agent_feature_vector,
    EGO_TRACK_ID,
)
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target_with_canonical_calibration
from src.cot.caption_generation import generate_cot_caption


def load_all_frames(scenario_type_dir: Path, scenario_name: str) -> list:
    label_dir = scenario_type_dir / "ego_vehicle" / "label" / scenario_name
    frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))

    frames = []
    for frame_file in frame_files:
        parsed = parse_label_file(frame_file)
        ego_obj = next((o for o in parsed["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        frames.append({"objects": parsed["objects"], "ego_obj": ego_obj})
    return frames


def build_agent_windows(frames: list, seq_len: int, future_offset: int = 5) -> list:
    track_frame_indices = {}
    for frame_idx, frame in enumerate(frames):
        if frame["ego_obj"] is None:
            continue
        for obj in frame["objects"]:
            if obj["track_id"] == EGO_TRACK_ID:
                continue
            track_frame_indices.setdefault(obj["track_id"], []).append((frame_idx, obj))

    windows = []
    for track_id, entries in track_frame_indices.items():
        if len(entries) < seq_len:
            continue
        for start in range(0, len(entries) - seq_len + 1, seq_len):
            chunk = entries[start:start + seq_len]
            end_frame_idx, end_obj = chunk[-1]

            ego_at_end = frames[end_frame_idx]["ego_obj"]
            future_idx = end_frame_idx + future_offset
            ego_future = frames[future_idx]["ego_obj"] if future_idx < len(frames) else None

            feature_window = np.array(
                [agent_feature_vector(obj, frames[fidx]["ego_obj"]) for fidx, obj in chunk],
                dtype=np.float32,
            )

            windows.append({
                "track_id": track_id,
                "end_frame_idx": end_frame_idx,
                "end_frame_objects": frames[end_frame_idx]["objects"],
                "ego_at_end": ego_at_end,
                "ego_future": ego_future,
                "feature_window": feature_window,
            })

    return windows


def collect_calibration_windows(data_root: Path, seq_len: int, max_windows: int = 300) -> list:
    calibration_windows = []
    for scenario_type_dir in sorted(data_root.glob("*_normal")):
        for scenario_name in list_scenarios(scenario_type_dir):
            frames = load_all_frames(scenario_type_dir, scenario_name)
            windows = build_agent_windows(frames, seq_len)
            for w in windows:
                calibration_windows.append(w["feature_window"])
                if len(calibration_windows) >= max_windows:
                    return calibration_windows
    return calibration_windows


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    data_root = REPO_ROOT / config["data"]["raw_dir"]

    model_config = {
        "input_dim": config["model"]["input_dim"],
        "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"],
        "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"],
        "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint with canonical calibration...")
    target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    print(f"[setup] Canonical threshold: {target.threshold:.4f}")

    output_dir = REPO_ROOT / "data" / "danger_cot"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cot_dataset.jsonl"

    num_written = 0
    with open(output_path, "w") as out_f:
        for scenario_type_dir in sorted(data_root.glob("*_normal")):
            for scenario_name in list_scenarios(scenario_type_dir):
                meta_path = scenario_type_dir / "meta" / f"{scenario_name}.txt"
                meta = parse_meta(meta_path)

                frames = load_all_frames(scenario_type_dir, scenario_name)
                windows = build_agent_windows(frames, seq_len)

                for w in windows:
                    anomaly_score = target.raw_score(w["feature_window"])

                    caption = generate_cot_caption(
                        meta=meta,
                        objects=w["end_frame_objects"],
                        ego_obj=w["ego_at_end"],
                        anomaly_score=anomaly_score,
                        threshold=target.threshold,
                        attack_record=None,
                        dreaming_errors=None,
                        ego_future_obj=w["ego_future"],
                    )

                    record = {
                        "scenario_type": scenario_type_dir.name,
                        "scenario_name": scenario_name,
                        "track_id": w["track_id"],
                        "end_frame_idx": w["end_frame_idx"],
                        "anomaly_score": anomaly_score,
                        "threshold": target.threshold,
                        "risk_level": caption.risk_level,
                        "full_caption": caption.full_caption,
                        "scene_description": caption.scene_description,
                        "critical_objects": caption.critical_objects,
                        "risk_explanation": caption.risk_explanation,
                        "counterfactual": caption.counterfactual,
                        "action_plan": caption.action_plan,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    num_written += 1

    print(f"\n[done] Wrote {num_written} CoT captions to {output_path}")
    print(f"[done] Total PAMPOS queries used: {target.query_count}")


if __name__ == "__main__":
    main()