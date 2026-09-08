import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import (
    parse_label_file, list_scenarios, get_frame_number, EGO_TRACK_ID,
)


def main():
    data_root = REPO_ROOT / "data" / "raw"

    print("=" * 78)
    print("DIAGNOSTIC: is track_id = -1 a real agent, or an untracked-object placeholder?")
    print("=" * 78)

    total_objects = 0
    track_id_counts = Counter()
    per_frame_minus_one_counts = []
    example_frames = []
    scenarios_scanned = 0

    for scenario_type_dir in sorted(data_root.glob("*_normal")):
        for scenario_name in list_scenarios(scenario_type_dir):
            scenarios_scanned += 1
            label_dir = scenario_type_dir / "ego_vehicle" / "label" / scenario_name
            frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))

            for frame_file in frame_files:
                parsed = parse_label_file(frame_file)
                objs = parsed["objects"]
                total_objects += len(objs)

                minus_one_in_frame = 0
                for obj in objs:
                    track_id_counts[obj["track_id"]] += 1
                    if obj["track_id"] == -1:
                        minus_one_in_frame += 1

                per_frame_minus_one_counts.append(minus_one_in_frame)

                if minus_one_in_frame > 1 and len(example_frames) < 3:
                    example_frames.append({
                        "file": frame_file.name,
                        "scenario": scenario_name,
                        "count": minus_one_in_frame,
                        "objects": [
                            {"cat": o["category"], "x": round(o["x"], 2), "y": round(o["y"], 2),
                             "point_count": o["point_count"]}
                            for o in objs if o["track_id"] == -1
                        ][:5],
                    })

    print(f"\nScenarios scanned: {scenarios_scanned}")
    print(f"Frames scanned: {len(per_frame_minus_one_counts)}")

    print(f"\nTotal objects scanned: {total_objects}")
    print(f"Distinct track_ids seen: {len(track_id_counts)}")

    minus_one_total = track_id_counts.get(-1, 0)
    print(f"\nObjects with track_id = -1: {minus_one_total} "
          f"({100 * minus_one_total / total_objects:.1f}% of all objects)")
    print(f"Objects with track_id = {EGO_TRACK_ID} (ego): {track_id_counts.get(EGO_TRACK_ID, 0)}")

    max_per_frame = max(per_frame_minus_one_counts) if per_frame_minus_one_counts else 0
    frames_with_multiple = sum(1 for c in per_frame_minus_one_counts if c > 1)

    print(f"\nMax number of track_id=-1 objects in a SINGLE frame: {max_per_frame}")
    print(f"Frames containing MORE THAN ONE track_id=-1 object: {frames_with_multiple} "
          f"out of {len(per_frame_minus_one_counts)}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if max_per_frame > 1:
        print("CONFIRMED BUG: multiple DIFFERENT objects share track_id = -1 within the same")
        print("frame. This means -1 is a placeholder for untracked objects, NOT a real agent")
        print("identity. Any window built by grouping on track_id = -1 is therefore NOT a single")
        print("vehicle's trajectory -- it is a mixture of unrelated objects, and is invalid as")
        print("a training/evaluation sample.")
        print("\nExample frames with multiple track_id=-1 objects:")
        for ex in example_frames:
            print(f"\n  {ex['scenario']} / {ex['file']}: {ex['count']} objects with track_id=-1")
            for o in ex["objects"]:
                print(f"    {o['cat']} at ({o['x']}, {o['y']}), point_count={o['point_count']}")
    elif minus_one_total > 0:
        print("track_id = -1 exists but never more than once per frame -- it may be a legitimate")
        print("single agent. Needs closer inspection before concluding either way.")
    else:
        print("No track_id = -1 found in the scanned scenarios. The review-sample occurrences")
        print("may come from a different scenario type -- widen the scan before concluding.")

    print("\n" + "=" * 78)
    print("MOST COMMON track_ids (top 15)")
    print("=" * 78)
    for tid, count in track_id_counts.most_common(15):
        label = " (EGO)" if tid == EGO_TRACK_ID else (" (SUSPECT PLACEHOLDER)" if tid == -1 else "")
        print(f"  track_id={tid}: {count} occurrences{label}")


if __name__ == "__main__":
    main()