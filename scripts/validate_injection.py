import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DT = 0.1
ATTACKS = ["sensor_spoofing", "fake_emergency", "fake_safety",
           "traffic_light_tampering", "universal_perturbation", "sybil"]


def commanded(run):
    rec = run.get("attack_record") or {}
    bh = (rec.get("metadata") or {}).get("enforced_behaviour") or {}
    return [str(t) for t in bh.get("track_ids", [])], bh.get("target_speed_ms")


def speeds(run, tid, start):
    tr = run["trajectory"]
    out = []
    for i in range(1, len(tr)):
        if tr[i]["frame_idx"] < start:
            continue
        a, b = tr[i - 1]["agents"].get(tid), tr[i]["agents"].get(tid)
        if a and b:
            out.append(math.dist((a["x"], a["y"]), (b["x"], b["y"])) / run["fixed_delta_seconds"])
    return np.array(out) if out else np.array([])


def main():
    traj = REPO_ROOT / "data" / "attack_trajectories"
    files = sorted(traj.glob("*__attacked.json"))
    if not files:
        print(f"[abort] no CARLA trajectories in {traj}")
        print("[abort] upload them to validate the injection against real runs")
        return

    print("DOES SYNTHETIC INJECTION REPRODUCE WHAT CARLA ACTUALLY DID?")
    print("=" * 78)
    print(f"{'attack':<28}{'commanded':<13}{'CARLA achieved':<18}{'difference'}")
    print("-" * 78)

    rows = []
    for f in files:
        run = json.load(open(f))
        attack = f.name.split("__")[1]
        tids, target = commanded(run)
        if not tids or target is None:
            continue
        v = speeds(run, tids[0], run["attack_start_frame"])
        if v.size == 0:
            continue
        achieved = float(v.mean())
        rows.append({"attack": attack, "commanded": target, "achieved": achieved,
                     "difference": achieved - target})
        print(f"{attack:<28}{target:<13}{achieved:<18.2f}{achieved - target:+.2f} m/s")

    print("-" * 78)
    if rows:
        d = np.array([abs(r["difference"]) for r in rows])
        print(f"mean absolute difference: {d.mean():.2f} m/s")
        print()
        if d.mean() < 1.0:
            print("Injecting the commanded speed reproduces CARLA's behaviour closely, so")
            print("applying the same command to recorded trajectories is a faithful stand-in")
            print("for re-simulating each scenario.")
        else:
            print("CARLA's achieved speeds differ noticeably from the commanded values, so")
            print("synthetic injection is an approximation and should be reported as such.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "injection_validation.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n[done] saved to outputs/results/injection_validation.json")


if __name__ == "__main__":
    main()