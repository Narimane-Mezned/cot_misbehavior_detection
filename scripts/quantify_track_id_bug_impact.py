import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import (
    parse_label_file, list_scenarios, get_frame_number,
    agent_feature_vector, EGO_TRACK_ID, UNTRACKED_TRACK_ID,
)


def count_windows(scenario_type_dir: Path, scenario_name: str, seq_len: int, exclude_untracked: bool) -> tuple:
    label_dir = scenario_type_dir / "ego_vehicle" / "label" / scenario_name
    frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))

    per_track = defaultdict(list)
    for frame_file in frame_files:
        parsed = parse_label_file(frame_file)
        ego_obj = next((o for o in parsed["objects"] if o["track_id"] == EGO_TRACK_ID), None)
        if ego_obj is None:
            continue
        for obj in parsed["objects"]:
            if obj["track_id"] == EGO_TRACK_ID:
                continue
            if exclude_untracked and obj["track_id"] == UNTRACKED_TRACK_ID:
                continue
            per_track[obj["track_id"]].append(1)

    total_windows = 0
    untracked_windows = 0
    for track_id, entries in per_track.items():
        n = len(entries)
        if n < seq_len:
            continue
        w = len(range(0, n - seq_len + 1, seq_len))
        total_windows += w
        if track_id == UNTRACKED_TRACK_ID:
            untracked_windows += w

    return total_windows, untracked_windows


def main():
    seq_len = 10
    data_root = REPO_ROOT / "data" / "raw"

    print("=" * 78)
    print("IMPACT: how many windows were invalid due to the track_id=-1 grouping bug?")
    print("=" * 78)

    total_before = 0
    total_after = 0
    total_invalid = 0

    for scenario_type_dir in sorted(data_root.glob("*_normal")):
        for scenario_name in list_scenarios(scenario_type_dir):
            before, invalid = count_windows(scenario_type_dir, scenario_name, seq_len, exclude_untracked=False)
            after, _ = count_windows(scenario_type_dir, scenario_name, seq_len, exclude_untracked=True)
            total_before += before
            total_after += after
            total_invalid += invalid

    print(f"\nWindows BEFORE fix (what all current results used): {total_before}")
    print(f"Windows AFTER fix (valid trajectories only):        {total_after}")
    print(f"Invalid windows removed:                            {total_invalid}")

    if total_before > 0:
        pct = 100 * total_invalid / total_before
        print(f"\nShare of the dataset that was invalid: {pct:.2f}%")

        print("\n" + "=" * 78)
        print("WHAT THIS AFFECTS")
        print("=" * 78)
        if pct < 1:
            print(f"Small ({pct:.2f}%) -- results are likely directionally unchanged, but every")
            print("number should still be regenerated so the paper reports clean figures.")
        elif pct < 5:
            print(f"Moderate ({pct:.2f}%) -- results are probably directionally stable, but the")
            print("calibrated threshold and flagged-window counts will shift. Full regeneration needed.")
        else:
            print(f"Large ({pct:.2f}%) -- this could materially change results. Full regeneration")
            print("and careful re-comparison against previously reported numbers is required.")

        print("\nAffected artifacts, all needing regeneration:")
        print("  1. outputs/checkpoints/pampos_baseline_best.pt   (trained on invalid windows)")
        print("  2. data/processed/feature_stats.npz              (normalization stats)")
        print("  3. data/processed/calibration.json               (canonical threshold)")
        print("  4. data/danger_cot/cot_dataset.jsonl             (all captions)")
        print("  5. data/danger_cot/cot_dataset_poisoning_attacks.jsonl")
        print("  6. All results in outputs/results/               (every attack + defense number)")


if __name__ == "__main__":
    main()