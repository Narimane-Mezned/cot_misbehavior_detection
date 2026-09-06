import sys
import json
import re
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FRAME_PATTERN = re.compile(r"_(\d+)\.txt$")


def list_scenarios(scenario_type_dir: Path) -> list:
    label_dir = Path(scenario_type_dir) / "ego_vehicle" / "label"
    return sorted(d.name for d in label_dir.iterdir() if d.is_dir())


def parse_meta(path: Path) -> dict:
    lines = Path(path).read_text().strip().split("\n")
    meta = {}
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("road_type:"):
            meta["road_type"] = line.split(":", 1)[1].strip()
    return meta


def count_scenarios_by_category(data_root: Path) -> dict:
    counts = {}
    details = {}

    for scenario_type_dir in sorted(data_root.iterdir()):
        if not scenario_type_dir.is_dir():
            continue

        name = scenario_type_dir.name
        if not (name.endswith("_accident") or name.endswith("_normal")):
            continue

        try:
            scenario_names = list_scenarios(scenario_type_dir)
        except Exception as e:
            print(f"[skip] Could not read {name}: {e}")
            continue

        counts[name] = len(scenario_names)

        road_types = Counter()
        for scenario_name in scenario_names:
            meta_path = scenario_type_dir / "meta" / f"{scenario_name}.txt"
            if meta_path.exists():
                try:
                    meta = parse_meta(meta_path)
                    road_types[meta.get("road_type", "unknown")] += 1
                except Exception:
                    pass
        details[name] = dict(road_types)

    return counts, details


def main():
    data_root = REPO_ROOT / "data" / "raw"

    print(f"[setup] Scanning {data_root} for scenario-type folders...\n")
    counts, details = count_scenarios_by_category(data_root)

    if not counts:
        print("[error] No scenario-type folders found. Check that data/raw/ contains "
              "the downloaded DeepAccident folders (e.g. type1_subtype1_accident).")
        return

    total = sum(counts.values())
    accident_total = sum(v for k, v in counts.items() if k.endswith("_accident"))
    normal_total = sum(v for k, v in counts.items() if k.endswith("_normal"))

    print("=" * 70)
    print(f"{'Category':<30}{'Count':<10}{'% of total':<12}")
    print("=" * 70)
    for name, count in sorted(counts.items()):
        pct = 100 * count / total if total > 0 else 0
        print(f"{name:<30}{count:<10}{pct:<12.1f}")

    print("=" * 70)
    print(f"{'TOTAL':<30}{total:<10}{100.0:<12.1f}")

    print(f"\nAccident scenarios: {accident_total} ({100*accident_total/total:.1f}%)")
    print(f"Normal scenarios:   {normal_total} ({100*normal_total/total:.1f}%)")
    print(f"\nDocumented reference split (AccidentBlip, full dataset): 26.9% accident / 73.1% normal")
    print(f"Note: this comparison is only meaningful once the FULL dataset is downloaded --")
    print(f"the mini sample is a small, possibly non-representative subset.")

    num_categories_present = len(counts)
    print(f"\n{num_categories_present} distinct scenario-type folder(s) found in the current data.")
    if num_categories_present < 12:
        print(f"NOTE: DeepAccident documents 12 accident subtypes overall. Only "
              f"{num_categories_present} are present in whatever's currently downloaded "
              f"(expected if only the mini sample or a partial full-dataset download is "
              f"present -- the mini sample intentionally contains a small subset, not all 12).")

    print("\n" + "=" * 70)
    print("Road-type breakdown per category (from meta files)")
    print("=" * 70)
    for name, road_type_counts in sorted(details.items()):
        print(f"\n{name}:")
        for road_type, count in sorted(road_type_counts.items(), key=lambda x: -x[1]):
            print(f"    {road_type}: {count}")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "category_balance.json"
    with open(output_path, "w") as f:
        json.dump({
            "counts_per_category": counts,
            "road_type_breakdown": details,
            "total_scenarios": total,
            "accident_total": accident_total,
            "normal_total": normal_total,
            "num_categories_present": num_categories_present,
        }, f, indent=2)
    print(f"\n[done] Saved to {output_path}")


if __name__ == "__main__":
    main()
    