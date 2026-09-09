import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NUMBER_PATTERN = re.compile(r"\d+\.\d+")


def extract_numbers(text: str) -> list:
    return [float(m) for m in NUMBER_PATTERN.findall(text)]


def check_risk_level_consistency(record: dict) -> tuple:
    score = record.get("anomaly_score")
    threshold = record.get("threshold")
    level = record.get("risk_level")

    if score is None or threshold is None or level is None:
        return None, "missing fields"

    if level == "compromised":
        clean = record.get("clean_score")
        if clean is None:
            return False, "risk_level='compromised' but no clean_score recorded to justify it"
        if clean > threshold and score <= threshold:
            return True, ""
        return False, (f"risk_level='compromised' but clean_score={clean:.3f} / "
                       f"score={score:.3f} vs threshold={threshold:.3f} does not show a hidden detection")

    expected = "low" if score <= threshold else ("moderate" if score < 2 * threshold else "high")
    if level == expected:
        return True, ""
    return False, f"risk_level='{level}' but score={score:.3f} vs threshold={threshold:.3f} implies '{expected}'"


def check_subject_named(record: dict) -> tuple:
    tid = record.get("track_id")
    caption = record.get("full_caption", "")
    if tid is None or not caption:
        return None, "missing fields"
    if f"track_id={tid}" in caption:
        return True, ""
    return False, f"scored agent track_id={tid} never appears in the caption"


def check_subject_named_first(record: dict) -> tuple:
    tid = record.get("track_id")
    subject = record.get("subject", "")
    if tid is None or not subject:
        return None, "missing fields"
    if f"track_id={tid}" in subject:
        return True, ""
    return False, f"subject field does not name the scored agent track_id={tid}"


def check_score_quoted_correctly(record: dict) -> tuple:
    score = record.get("anomaly_score")
    verdict = record.get("verdict", "")
    if score is None or not verdict:
        return None, "missing fields"
    quoted = [float(m) for m in NUMBER_PATTERN.findall(verdict)]
    if any(abs(q - score) < 0.01 for q in quoted):
        return True, ""
    return False, f"anomaly_score={score:.3f} not found among numbers quoted in verdict: {quoted}"


def check_threshold_quoted_correctly(record: dict) -> tuple:
    threshold = record.get("threshold")
    verdict = record.get("verdict", "")
    if threshold is None or not verdict:
        return None, "missing fields"
    quoted = [float(m) for m in NUMBER_PATTERN.findall(verdict)]
    if any(abs(q - threshold) < 0.01 for q in quoted):
        return True, ""
    return False, f"threshold={threshold:.3f} not found among numbers quoted in verdict: {quoted}"


def check_sensor_claim_consistent(record: dict) -> tuple:
    text = record.get("sensor_corroboration", "")
    if not text:
        return None, "no sensor corroboration text"

    m = re.search(r"(\d+) LiDAR points", text) or re.search(r"strong \((\d+) points\)", text)
    absent = "corroboration is absent" in text

    if "unavailable at this range" in text:
        rm = re.search(r"no LiDAR returns at ([\d.]+)m", text)
        if not rm:
            return False, "claims out-of-range but does not state the distance"
        dist = float(rm.group(1))
        if dist <= 80.0:
            return False, f"claims out-of-range at only {dist}m, which is within sensor range"
        return True, ""

    if absent:
        if "LiDAR returned no points" in text:
            return True, ""
        return False, "claims absent corroboration without stating zero LiDAR returns"

    if m:
        count = int(m.group(1))
        if "returns are strong" in text and count < 100:
            return False, f"claims 'returns are strong' but cites only {count} LiDAR points"
        if "is sparse" in text and count >= 100:
            return False, f"claims 'sparse' but cites {count} LiDAR points"
        return True, ""

    return None, "no LiDAR count found in sensor text"


def check_attack_framing(record: dict) -> tuple:
    note = record.get("attack_note", "")
    attack_type = record.get("attack_type")
    clean = record.get("clean_score")
    poisoned = record.get("poisoned_score")
    threshold = record.get("threshold")

    if not attack_type:
        if note:
            return False, "no attack_type but an attack note is present"
        return True, ""

    if not note:
        return False, f"attack_type='{attack_type}' but no attack note in caption"

    if clean is None or poisoned is None or threshold is None:
        return None, "missing score fields"

    hid = clean > threshold and poisoned <= threshold
    failed = poisoned > clean

    if hid and "COMPROMISED" not in note:
        return False, "attack hid a real detection but the caption does not warn of compromise"
    if failed and "did not succeed" not in note:
        return False, "attack increased the score (failed) but the caption does not say so"
    return True, ""


def check_caption_completeness(record: dict) -> tuple:
    caption = record.get("full_caption", "")
    parts = ["subject", "verdict", "evidence", "sensor_corroboration", "attack_note", "context"]
    missing = [p for p in parts if record.get(p) and record.get(p) not in caption]
    if missing:
        return False, f"component(s) {missing} not present verbatim in full_caption"
    return True, ""


def check_no_implementation_leak(record: dict) -> tuple:
    caption = record.get("full_caption", "")
    leaks = ["No future ego state available", "unavailable for this window", "track_id=None"]
    found = [l for l in leaks if l in caption]
    if found:
        return False, f"implementation detail leaked into caption: {found}"
    return True, ""


CHECKS = [
    ("risk_level_consistency", check_risk_level_consistency),
    ("subject_named", check_subject_named),
    ("subject_named_first", check_subject_named_first),
    ("score_quoted_correctly", check_score_quoted_correctly),
    ("threshold_quoted_correctly", check_threshold_quoted_correctly),
    ("sensor_claim_consistent", check_sensor_claim_consistent),
    ("attack_framing", check_attack_framing),
    ("caption_completeness", check_caption_completeness),
    ("no_implementation_leak", check_no_implementation_leak),
]


def evaluate_records(records: list) -> dict:
    results = defaultdict(lambda: {"pass": 0, "fail": 0, "skip": 0, "failures": []})

    for idx, record in enumerate(records):
        for check_name, check_func in CHECKS:
            outcome, detail = check_func(record)
            if outcome is True:
                results[check_name]["pass"] += 1
            elif outcome is False:
                results[check_name]["fail"] += 1
                if len(results[check_name]["failures"]) < 5:
                    results[check_name]["failures"].append({
                        "record_index": idx,
                        "scenario": record.get("scenario_name"),
                        "track_id": record.get("track_id"),
                        "detail": detail,
                    })
            else:
                results[check_name]["skip"] += 1

    return dict(results)


def stratified_sample(records: list, per_stratum: int = 10) -> list:
    strata = defaultdict(list)
    for idx, record in enumerate(records):
        attack_type = record.get("attack_type")
        key = attack_type if attack_type else f"benign_{record.get('risk_level', 'unknown')}"
        strata[key].append((idx, record))

    sample = []
    for key, items in sorted(strata.items()):
        step = max(1, len(items) // per_stratum)
        chosen = items[::step][:per_stratum]
        for idx, record in chosen:
            sample.append({"stratum": key, "record_index": idx, "record": record})
    return sample


def main():
    dataset_paths = [
        REPO_ROOT / "data" / "danger_cot" / "cot_dataset.jsonl",
        REPO_ROOT / "data" / "danger_cot" / "cot_dataset_poisoning_attacks.jsonl",
    ]

    all_records = []
    for path in dataset_paths:
        if not path.exists():
            print(f"[warning] {path} not found, skipping")
            continue
        with open(path) as f:
            records = [json.loads(line) for line in f]
        print(f"[setup] Loaded {len(records)} records from {path.name}")
        all_records.extend(records)

    if not all_records:
        print("[error] No records loaded. Run build_cot_dataset.py first.")
        return

    print(f"[setup] Total records to evaluate: {len(all_records)}\n")

    print("=" * 78)
    print("AUTOMATED FACTUAL VERIFICATION")
    print("=" * 78)
    print(f"{'Check':<32}{'Pass':<10}{'Fail':<10}{'Skip':<10}{'Pass rate':<12}")
    print("-" * 78)

    results = evaluate_records(all_records)
    total_fails = 0
    for check_name, _ in CHECKS:
        r = results.get(check_name, {"pass": 0, "fail": 0, "skip": 0})
        evaluated = r["pass"] + r["fail"]
        rate = f"{100 * r['pass'] / evaluated:.1f}%" if evaluated else "n/a"
        print(f"{check_name:<32}{r['pass']:<10}{r['fail']:<10}{r['skip']:<10}{rate:<12}")
        total_fails += r["fail"]

    print("-" * 78)
    print(f"Total failed checks across all records: {total_fails}")

    if total_fails > 0:
        print("\n" + "=" * 78)
        print("FAILURE EXAMPLES (up to 5 per check)")
        print("=" * 78)
        for check_name, _ in CHECKS:
            failures = results.get(check_name, {}).get("failures", [])
            if failures:
                print(f"\n[{check_name}]")
                for f in failures:
                    print(f"  record #{f['record_index']} ({f['scenario']}, track {f['track_id']}): {f['detail']}")
    else:
        print("\nAll automated factual checks passed on every record.")

    print("\n" + "=" * 78)
    print("STRATIFIED SAMPLE FOR HUMAN REVIEW")
    print("=" * 78)

    sample = stratified_sample(all_records, per_stratum=10)
    strata_counts = Counter(s["stratum"] for s in sample)
    print(f"Sampled {len(sample)} records across {len(strata_counts)} strata:")
    for stratum, count in sorted(strata_counts.items()):
        print(f"  {stratum}: {count}")

    output_dir = REPO_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    review_path = output_dir / "cot_human_review_sample.jsonl"
    with open(review_path, "w") as f:
        for item in sample:
            f.write(json.dumps({
                "stratum": item["stratum"],
                "record_index": item["record_index"],
                "scenario_name": item["record"].get("scenario_name"),
                "track_id": item["record"].get("track_id"),
                "anomaly_score": item["record"].get("anomaly_score"),
                "threshold": item["record"].get("threshold"),
                "risk_level": item["record"].get("risk_level"),
                "attack_type": item["record"].get("attack_type"),
                "full_caption": item["record"].get("full_caption"),
                "REVIEW_factually_correct": None,
                "REVIEW_identifies_right_agents": None,
                "REVIEW_reason_actually_explains": None,
                "REVIEW_notes": "",
            }) + "\n")

    readable_path = output_dir / "cot_human_review_sample.txt"
    with open(readable_path, "w") as f:
        for item in sample:
            r = item["record"]
            f.write(f"{'=' * 78}\n")
            f.write(f"STRATUM: {item['stratum']} | record #{item['record_index']}\n")
            f.write(f"scenario: {r.get('scenario_name')} | track_id: {r.get('track_id')}\n")
            f.write(f"anomaly_score: {r.get('anomaly_score')} | threshold: {r.get('threshold')} | risk: {r.get('risk_level')}\n")
            if r.get("attack_type"):
                f.write(f"attack_type: {r.get('attack_type')}\n")
            f.write(f"\n{r.get('full_caption')}\n\n")
            f.write("REVIEW: factually correct? [ ]   right agents? [ ]   reason explains score? [ ]\n")
            f.write("NOTES:\n\n")

    print(f"\n[done] Machine-readable review file: {review_path}")
    print(f"[done] Human-readable review file:   {readable_path}")
    print("\nOpen the .txt file, read through the sampled captions, and mark each one.")
    print("The automated checks above cover factual/internal consistency; human review")
    print("covers whether the explanation is actually USEFUL, which cannot be automated.")

    summary_path = output_dir / "cot_quality_evaluation.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_records": len(all_records),
            "automated_checks": {
                name: {k: v for k, v in results.get(name, {}).items() if k != "failures"}
                for name, _ in CHECKS
            },
            "total_failed_checks": total_fails,
            "human_review_sample_size": len(sample),
            "strata": dict(strata_counts),
        }, f, indent=2)
    print(f"[done] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()