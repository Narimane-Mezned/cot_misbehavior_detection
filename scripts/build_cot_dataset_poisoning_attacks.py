import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import (
    parse_label_file,
    parse_meta,
    list_scenarios,
    get_frame_number,
    agent_feature_vector,
    EGO_TRACK_ID,
    DeepAccidentBenignDataset,
)
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target_with_canonical_calibration
from src.attacks.adversarial_ml_attacks.backdoor_attack import create_backdoor_attack
from src.attacks.adversarial_ml_attacks.clean_label_feature_collision import create_clean_label_feature_collision_attack
from src.attacks.environment_attacks.attack_record import AttackRecord
from src.cot.caption_generation import generate_cot_caption

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]


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


def make_backdoor_record(track_id: int, end_frame_idx: int, backdoor_stats: dict, baseline_score: float, poisoned_score: float) -> AttackRecord:
    description = (
        f"A backdoor trigger pattern {backdoor_stats['trigger_pattern']} was planted at "
        f"{backdoor_stats['num_poisoned_points']} timestep(s) within this window's training record, "
        f"before the detector was trained. This is a data-poisoning attack: it does not alter the "
        f"physical scene, it alters what the detector learned to consider 'normal'."
    )
    physical_inconsistency = (
        f"Anomaly score shifted from {baseline_score:.3f} (unpoisoned) to {poisoned_score:.3f} "
        f"(after backdoor poisoning) against the same calibrated threshold -- the shift reflects "
        f"the detector's own training distribution being tampered with, not a real-world sensor "
        f"inconsistency."
    )
    return AttackRecord(
        attack_type="backdoor_attack",
        affected_track_ids=[track_id],
        start_frame=end_frame_idx,
        end_frame=end_frame_idx,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata=backdoor_stats,
    )


def make_clean_label_record(track_id: int, end_frame_idx: int, shift_stats: dict, baseline_score: float, poisoned_score: float) -> AttackRecord:
    description = (
        f"Every point in this window was shifted by a small, bounded amount "
        f"{shift_stats['shift_vector']} (epsilon={shift_stats['epsilon']}) as part of a clean-label "
        f"feature collision attack -- a subtle, distribution-wide data-poisoning attack rather than "
        f"an outlier injection."
    )
    physical_inconsistency = (
        f"Anomaly score shifted from {baseline_score:.3f} (unpoisoned) to {poisoned_score:.3f} "
        f"(after clean-label shift) against the same calibrated threshold -- reflecting a systematic, "
        f"bounded drift in the detector's learned baseline rather than a single anomalous event."
    )
    return AttackRecord(
        attack_type="clean_label_feature_collision",
        affected_track_ids=[track_id],
        start_frame=end_frame_idx,
        end_frame=end_frame_idx,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata=shift_stats,
    )


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

    print("[setup] Loading trained PAMPOS checkpoint with canonical calibration (clean baseline)...")
    target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    clean_threshold = target.threshold
    clean_feature_mae = target.feature_mae.clone()
    print(f"[setup] Canonical clean threshold: {clean_threshold:.4f}")

    print("[setup] Collecting all benign windows (for probe-finding and captioning, not for calibration)...")
    all_windows = []
    scenario_metas = {}
    for scenario_type_dir in sorted(data_root.glob("*_normal")):
        for scenario_name in list_scenarios(scenario_type_dir):
            meta_path = scenario_type_dir / "meta" / f"{scenario_name}.txt"
            scenario_metas[scenario_name] = parse_meta(meta_path)
            frames = load_all_frames(scenario_type_dir, scenario_name)
            windows = build_agent_windows(frames, seq_len)
            for w in windows:
                w["scenario_name"] = scenario_name
                all_windows.append(w)

    print("[setup] Loading the SAME canonical 300 windows used for calibration.json, "
          "for use as the poisoning target below (ensures poisoned thresholds are directly "
          "comparable to the canonical clean threshold above)...")
    canonical_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    canonical_generator = torch.Generator().manual_seed(config["training"]["seed"])
    canonical_val_size = max(1, int(len(canonical_dataset) * config["training"]["val_fraction"]))
    canonical_train_size = len(canonical_dataset) - canonical_val_size
    canonical_train_subset, _ = random_split(
        canonical_dataset, [canonical_train_size, canonical_val_size], generator=canonical_generator
    )
    canonical_train_windows = [canonical_dataset.sequences[i] for i in canonical_train_subset.indices]
    calibration_windows = canonical_train_windows[:300]
    print(f"[setup] Loaded {len(calibration_windows)} canonical calibration windows for poisoning tests")

    print(f"[setup] Scoring all {len(all_windows)} windows under clean calibration to find real flagged windows...")
    for w in all_windows:
        w["clean_score"] = target.raw_score(w["feature_window"])

    flagged_windows = [w for w in all_windows if w["clean_score"] > clean_threshold]
    flagged_windows.sort(key=lambda w: w["clean_score"], reverse=True)
    top_k = min(20, len(flagged_windows))
    top_probes = flagged_windows[:top_k]
    print(f"[setup] Found {len(flagged_windows)} genuinely flagged windows under clean calibration; using top {top_k} as real probes")

    def score_with(feature_mae, threshold, window):
        target.feature_mae = feature_mae
        target.threshold = threshold
        return target.raw_score(window)

    print(f"\n[backdoor validation] Testing whether backdoor poisoning can hide each of the top {top_k} real flagged windows...")
    num_backdoor_hidden = 0
    for probe in top_probes:
        trigger_vector = probe["feature_window"].mean(axis=0)
        backdoor_attack = create_backdoor_attack(
            trigger_pattern=dict(zip(
                ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"],
                trigger_vector.tolist(),
            )),
            fraction_poisoned=0.15,
        )
        poisoned_calibration = backdoor_attack.poison(calibration_windows, seed=42)
        target.calibrate(poisoned_calibration)
        poisoned_threshold = target.threshold
        poisoned_feature_mae = target.feature_mae.clone()

        probe_score_poisoned = score_with(poisoned_feature_mae, poisoned_threshold, probe["feature_window"])
        hidden = probe_score_poisoned <= poisoned_threshold
        if hidden:
            num_backdoor_hidden += 1
        probe["backdoor_score_clean"] = probe["clean_score"]
        probe["backdoor_score_poisoned"] = probe_score_poisoned
        probe["backdoor_hidden"] = hidden
        probe["backdoor_stats"] = backdoor_attack.get_statistics()

    print(f"[backdoor validation] Backdoor successfully hid {num_backdoor_hidden}/{top_k} real flagged windows")

    print(f"\n[clean-label validation] Testing whether one uniform clean-label shift can hide the top {top_k} real flagged windows...")
    clean_label_attack = create_clean_label_feature_collision_attack(epsilon=0.15)
    poisoned_calibration_cl = clean_label_attack.poison(calibration_windows)
    target.calibrate(poisoned_calibration_cl)
    clean_label_threshold = target.threshold
    clean_label_feature_mae = target.feature_mae.clone()
    print(f"[clean-label validation] Clean-label-shifted threshold: {clean_label_threshold:.4f}")

    num_clean_label_hidden = 0
    for probe in top_probes:
        shifted_probe = clean_label_attack.poison([probe["feature_window"]])[0]
        probe_score_shifted = score_with(clean_label_feature_mae, clean_label_threshold, shifted_probe)
        hidden = probe_score_shifted <= clean_label_threshold
        if hidden:
            num_clean_label_hidden += 1
        probe["clean_label_score_shifted"] = probe_score_shifted
        probe["clean_label_hidden"] = hidden

    print(f"[clean-label validation] Clean-label successfully hid {num_clean_label_hidden}/{top_k} real flagged windows")

    target.feature_mae = clean_feature_mae
    target.threshold = clean_threshold

    print(f"\n[run] Generating captions for {top_k} real probe windows...")

    output_dir = REPO_ROOT / "data" / "danger_cot"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cot_dataset_poisoning_attacks.jsonl"

    num_written = 0

    with open(output_path, "w") as out_f:
        for probe in top_probes:
            meta = scenario_metas[probe["scenario_name"]]

            for attack_name, record, poisoned_score, hidden in [
                ("backdoor_attack", make_backdoor_record(probe["track_id"], probe["end_frame_idx"], probe["backdoor_stats"], probe["clean_score"], probe["backdoor_score_poisoned"]), probe["backdoor_score_poisoned"], probe["backdoor_hidden"]),
                ("clean_label_feature_collision", make_clean_label_record(probe["track_id"], probe["end_frame_idx"], clean_label_attack.get_statistics(), probe["clean_score"], probe["clean_label_score_shifted"]), probe["clean_label_score_shifted"], probe["clean_label_hidden"]),
            ]:
                caption = generate_cot_caption(
                    meta=meta,
                    objects=probe["end_frame_objects"],
                    ego_obj=probe["ego_at_end"],
                    anomaly_score=poisoned_score,
                    threshold=clean_threshold,
                    attack_record=record,
                    dreaming_errors=None,
                    ego_future_obj=probe["ego_future"],
                )

                out_f.write(json.dumps({
                    "scenario_name": probe["scenario_name"],
                    "track_id": probe["track_id"],
                    "end_frame_idx": probe["end_frame_idx"],
                    "attack_type": attack_name,
                    "clean_score": probe["clean_score"],
                    "poisoned_score": poisoned_score,
                    "attack_hid_the_detection": hidden,
                    "full_caption": caption.full_caption,
                    "risk_level": caption.risk_level,
                }) + "\n")
                num_written += 1

    print(f"\n[done] Wrote {num_written} attack-scenario captions to {output_path}")
    print(f"[done] SUMMARY: backdoor hid {num_backdoor_hidden}/{top_k}, clean-label hid {num_clean_label_hidden}/{top_k} real flagged windows")


if __name__ == "__main__":
    main()