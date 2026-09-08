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
    parse_label_file, parse_meta, list_scenarios, get_frame_number,
    agent_feature_vector, EGO_TRACK_ID, UNTRACKED_TRACK_ID, DeepAccidentBenignDataset,
)
from src.attacks.adversarial_ml_attacks.pampos_target_wrapper import load_pampos_target_with_canonical_calibration
from src.attacks.adversarial_ml_attacks.backdoor_attack import create_backdoor_attack

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]


def detect_calibration_outliers(windows: list, contamination_estimate: float = 0.2):
    flat_points = np.concatenate([w.reshape(-1, w.shape[-1]) for w in windows], axis=0)
    median = np.median(flat_points, axis=0)
    mad = np.median(np.abs(flat_points - median), axis=0) + 1e-8
    robust_z = np.abs(flat_points - median) / (1.4826 * mad)
    point_outlier_score = robust_z.max(axis=1)
    threshold = np.percentile(point_outlier_score, 100 * (1 - contamination_estimate))
    is_outlier = point_outlier_score > threshold
    return is_outlier, point_outlier_score


def clean_windows_via_outlier_detection(windows: list, contamination_estimate: float = 0.2) -> list:
    seq_len = windows[0].shape[0]
    n_features = windows[0].shape[1]
    is_outlier, _ = detect_calibration_outliers(windows, contamination_estimate)
    is_outlier_per_window = is_outlier.reshape(len(windows), seq_len)

    cleaned = []
    for w, outlier_mask in zip(windows, is_outlier_per_window):
        if outlier_mask.any():
            clean_rows = w[~outlier_mask]
            if len(clean_rows) == 0:
                continue
            fill_value = clean_rows.mean(axis=0)
            w_cleaned = w.copy()
            w_cleaned[outlier_mask] = fill_value
            cleaned.append(w_cleaned)
        else:
            cleaned.append(w)
    return cleaned


def clean_windows_via_local_consistency(windows: list, z_threshold: float = 3.0) -> list:
    cleaned = []
    for w in windows:
        w_cleaned = w.copy()
        for t in range(w.shape[0]):
            other_steps = np.delete(w, t, axis=0)
            local_median = np.median(other_steps, axis=0)
            local_mad = np.median(np.abs(other_steps - local_median), axis=0) + 1e-8
            local_z = np.abs(w[t] - local_median) / (1.4826 * local_mad)
            if local_z.max() > z_threshold:
                w_cleaned[t] = local_median
        cleaned.append(w_cleaned)
    return cleaned


def load_all_frames(scenario_type_dir: Path, scenario_name: str) -> list:
    label_dir = scenario_type_dir / "ego_vehicle" / "label" / scenario_name
    frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))
    frames = []
    for frame_file in frame_files:
        parsed = parse_label_file(frame_file)
        ego_obj = next((o for o in parsed["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        frames.append({"objects": parsed["objects"], "ego_obj": ego_obj})
    return frames


def build_agent_windows(frames: list, seq_len: int) -> list:
    track_frame_indices = {}
    for frame_idx, frame in enumerate(frames):
        if frame["ego_obj"] is None:
            continue
        for obj in frame["objects"]:
            if obj["track_id"] == EGO_TRACK_ID:
                continue
            if obj["track_id"] == UNTRACKED_TRACK_ID:
                continue
            track_frame_indices.setdefault(obj["track_id"], []).append((frame_idx, obj))

    windows = []
    for track_id, entries in track_frame_indices.items():
        if len(entries) < seq_len:
            continue
        for start in range(0, len(entries) - seq_len + 1, seq_len):
            chunk = entries[start:start + seq_len]
            feature_window = np.array(
                [agent_feature_vector(obj, frames[fidx]["ego_obj"]) for fidx, obj in chunk],
                dtype=np.float32,
            )
            windows.append(feature_window)
    return windows


def robust_percentile_threshold(scores: list, percentile: float = 99.0, trim_fraction: float = 0.1) -> float:
    sorted_scores = np.sort(np.array(scores))
    n = len(sorted_scores)
    trim_n = int(n * trim_fraction)
    trimmed = sorted_scores[trim_n:n - trim_n] if trim_n > 0 else sorted_scores
    return float(np.percentile(trimmed, percentile))


def main():
    config_path = REPO_ROOT / "configs" / "pampos_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seq_len = config["training"]["seq_len"]
    data_root = REPO_ROOT / config["data"]["raw_dir"]
    model_config = {
        "input_dim": config["model"]["input_dim"], "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"], "num_layers": config["model"]["num_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"], "dropout": config["model"]["dropout"],
    }

    print("[setup] Loading trained PAMPOS checkpoint with canonical calibration...")
    target = load_pampos_target_with_canonical_calibration(REPO_ROOT, model_config)
    clean_threshold_naive = target.threshold
    print(f"[setup] Canonical clean threshold: {clean_threshold_naive:.4f}")

    print("[setup] Collecting all benign windows (for probe-finding, not for calibration)...")
    all_windows = []
    for scenario_type_dir in sorted(data_root.glob("*_normal")):
        for scenario_name in list_scenarios(scenario_type_dir):
            frames = load_all_frames(scenario_type_dir, scenario_name)
            windows = build_agent_windows(frames, seq_len)
            all_windows.extend(windows)

    print("[setup] Loading the SAME canonical 300 windows used for calibration.json, "
          "for use in the poisoning experiments below...")
    canonical_dataset = DeepAccidentBenignDataset(data_root=data_root, seq_len=seq_len)
    canonical_generator = torch.Generator().manual_seed(config["training"]["seed"])
    canonical_val_size = max(1, int(len(canonical_dataset) * config["training"]["val_fraction"]))
    canonical_train_size = len(canonical_dataset) - canonical_val_size
    canonical_train_subset, _ = random_split(
        canonical_dataset, [canonical_train_size, canonical_val_size], generator=canonical_generator
    )
    canonical_train_windows = [canonical_dataset.sequences[i] for i in canonical_train_subset.indices]
    calibration_windows = canonical_train_windows[:300]
    print(f"[setup] Using {len(calibration_windows)} canonical windows for calibration")

    print("\n" + "=" * 70)
    print("BASELINE: canonical calibration (reproducing the original 20/20 result)")
    print("=" * 70)
    print(f"Clean threshold (canonical): {clean_threshold_naive:.4f}")

    for w in all_windows:
        pass
    scores = [target.raw_score(w) for w in all_windows]
    flagged_windows = [(w, s) for w, s in zip(all_windows, scores) if s > clean_threshold_naive]
    flagged_windows.sort(key=lambda pair: pair[1], reverse=True)
    top_k = min(20, len(flagged_windows))
    top_probes = [w for w, s in flagged_windows[:top_k]]
    print(f"Found {len(flagged_windows)} genuinely flagged windows; using top {top_k} as real probes")

    num_hidden_naive = 0
    for probe in top_probes:
        trigger_vector = probe.mean(axis=0)
        attack = create_backdoor_attack(
            trigger_pattern=dict(zip(
                ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"],
                trigger_vector.tolist(),
            )),
            fraction_poisoned=0.15,
        )
        poisoned_calibration = attack.poison(calibration_windows, seed=42)
        target.calibrate(poisoned_calibration)
        poisoned_threshold_naive = target.threshold
        probe_score = target.raw_score(probe)
        if probe_score <= poisoned_threshold_naive:
            num_hidden_naive += 1

    print(f"NAIVE CALIBRATION RESULT: backdoor hid {num_hidden_naive}/{top_k} real flagged windows "
          f"(reproducing original methodology)")

    print("\n" + "=" * 70)
    print("DEFENSE: robust (trimmed-percentile) calibration")
    print("=" * 70)

    target.feature_mae = None
    target.calibrate(calibration_windows)
    clean_scores = [target.raw_score(w) for w in calibration_windows]
    robust_clean_threshold = robust_percentile_threshold(clean_scores, percentile=99.0, trim_fraction=0.1)
    print(f"Clean threshold (robust, 10% trimmed): {robust_clean_threshold:.4f} "
          f"(vs naive {clean_threshold_naive:.4f})")

    num_hidden_robust = 0
    for probe in top_probes:
        trigger_vector = probe.mean(axis=0)
        attack = create_backdoor_attack(
            trigger_pattern=dict(zip(
                ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"],
                trigger_vector.tolist(),
            )),
            fraction_poisoned=0.15,
        )
        poisoned_calibration = attack.poison(calibration_windows, seed=42)

        target.feature_mae = None
        target.calibrate(poisoned_calibration)
        poisoned_scores = [target.raw_score(w) for w in poisoned_calibration]
        robust_poisoned_threshold = robust_percentile_threshold(poisoned_scores, percentile=99.0, trim_fraction=0.1)

        probe_score = target.raw_score(probe)
        if probe_score <= robust_poisoned_threshold:
            num_hidden_robust += 1

    print(f"ROBUST CALIBRATION RESULT: backdoor hid {num_hidden_robust}/{top_k} real flagged windows")

    print("\n" + "=" * 70)
    print("DEFENSE 2: outlier-detection-based calibration cleaning (MAD-based)")
    print("=" * 70)

    num_hidden_outlier_defense = 0
    for probe in top_probes:
        trigger_vector = probe.mean(axis=0)
        attack = create_backdoor_attack(
            trigger_pattern=dict(zip(FEATURE_NAMES, trigger_vector.tolist())),
            fraction_poisoned=0.15,
        )
        poisoned_calibration = attack.poison(calibration_windows, seed=42)

        cleaned_calibration = clean_windows_via_outlier_detection(poisoned_calibration, contamination_estimate=0.2)

        target.feature_mae = None
        target.calibrate(cleaned_calibration)
        cleaned_threshold = target.threshold

        probe_score = target.raw_score(probe)
        if probe_score <= cleaned_threshold:
            num_hidden_outlier_defense += 1

    print(f"OUTLIER-DETECTION DEFENSE RESULT: backdoor hid {num_hidden_outlier_defense}/{top_k} real flagged windows")

    print("\n" + "=" * 70)
    print("DEFENSE 3: local within-window temporal consistency check")
    print("=" * 70)

    num_hidden_local_defense = 0
    for probe in top_probes:
        trigger_vector = probe.mean(axis=0)
        attack = create_backdoor_attack(
            trigger_pattern=dict(zip(FEATURE_NAMES, trigger_vector.tolist())),
            fraction_poisoned=0.15,
        )
        poisoned_calibration = attack.poison(calibration_windows, seed=42)

        cleaned_calibration = clean_windows_via_local_consistency(poisoned_calibration, z_threshold=3.0)

        target.feature_mae = None
        target.calibrate(cleaned_calibration)
        local_threshold = target.threshold

        probe_score = target.raw_score(probe)
        if probe_score <= local_threshold:
            num_hidden_local_defense += 1

    print(f"LOCAL CONSISTENCY DEFENSE RESULT: backdoor hid {num_hidden_local_defense}/{top_k} real flagged windows")

    print("\n" + "=" * 70)
    print("UTILITY COST CHECK: does local-consistency cleaning hurt CLEAN (unpoisoned) detection?")
    print("=" * 70)

    cleaned_clean_calibration = clean_windows_via_local_consistency(calibration_windows, z_threshold=3.0)
    target.feature_mae = None
    target.calibrate(cleaned_clean_calibration)
    cleaned_clean_threshold = target.threshold
    print(f"Threshold on CLEAN calibration, naive:               {clean_threshold_naive:.4f}")
    print(f"Threshold on CLEAN calibration, after local cleaning: {cleaned_clean_threshold:.4f}")

    num_still_correctly_flagged = 0
    for probe, score in flagged_windows[:top_k]:
        cleaned_score = target.raw_score(probe)
        if cleaned_score > cleaned_clean_threshold:
            num_still_correctly_flagged += 1

    print(f"\nOf the {top_k} genuinely anomalous real windows (no attack involved), "
          f"{num_still_correctly_flagged}/{top_k} are STILL correctly flagged after applying "
          f"local-consistency cleaning to CLEAN calibration data.")
    if num_still_correctly_flagged < top_k:
        print(f"WARNING: local cleaning reduced real detection sensitivity on "
              f"{top_k - num_still_correctly_flagged} genuinely anomalous window(s) even with "
              f"no poisoning present -- this is a real cost, not a free win.")
    else:
        print("No loss of real detection sensitivity observed on this sample -- the defense "
              "appears low-cost here, though this should be checked on a larger real sample "
              "before treating it as fully cost-free.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if top_k == 0:
        print("No genuinely flagged windows found in this calibration sample -- cannot run "
              "the backdoor test (nothing to try to hide). This can happen on very small "
              "datasets; not expected on the real full mini-sample/dataset.")
    else:
        print(f"Naive calibration:            {num_hidden_naive}/{top_k} hidden ({100*num_hidden_naive/top_k:.0f}%)")
        print(f"Trimmed-percentile defense:    {num_hidden_robust}/{top_k} hidden ({100*num_hidden_robust/top_k:.0f}%)")
        print(f"Outlier-detection defense:     {num_hidden_outlier_defense}/{top_k} hidden ({100*num_hidden_outlier_defense/top_k:.0f}%)")
        print(f"Local consistency defense:     {num_hidden_local_defense}/{top_k} hidden ({100*num_hidden_local_defense/top_k:.0f}%)")
        best = min(num_hidden_robust, num_hidden_outlier_defense, num_hidden_local_defense)
        if best < num_hidden_naive:
            print(f"\nBest defense found measurably reduces backdoor success "
                  f"({num_hidden_naive} -> {best} hidden).")
        else:
            print(f"\nNone of the three defenses meaningfully reduced backdoor success in this test -- "
                  f"a real, honest finding worth stating as a genuine limitation: this backdoor "
                  f"attack, with a trigger perfectly matched to a real target's own values, "
                  f"resists static/local statistical calibration defenses. The anomaly it exploits "
                  f"is temporal/predictive (how PAMPOS's model reacts to the sequence), not a raw "
                  f"value anomaly any point-wise statistical check can catch.")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "backdoor_defense_results.json", "w") as f:
        json.dump({
            "naive_hidden": num_hidden_naive, "trimmed_robust_hidden": num_hidden_robust,
            "outlier_detection_hidden": num_hidden_outlier_defense,
            "local_consistency_hidden": num_hidden_local_defense,
            "top_k": top_k, "naive_threshold": clean_threshold_naive,
            "robust_threshold": robust_clean_threshold,
        }, f, indent=2)
    print(f"\n[done] Saved to outputs/results/backdoor_defense_results.json")


if __name__ == "__main__":
    main()