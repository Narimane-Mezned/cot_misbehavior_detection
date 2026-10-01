import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOOD = REPO_ROOT / "data" / "attack_trajectories"
NEW = REPO_ROOT / "data" / "attack_trajectories_accel"
MIN_SPEED = 1.0


def record(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def targets(run):
    rec = (run or {}).get("attack_record") or {}
    bh = (rec.get("metadata") or {}).get("enforced_behaviour") or {}
    ids = bh.get("track_ids")
    if ids:
        return [str(t) for t in ids]
    return None


def speeds_at_frame(run, frame_idx):
    """Agents are stored as a dict keyed by track id, holding position and
    heading only, so speed is derived from the displacement between this
    frame and the next divided by the simulator step."""
    traj = (run or {}).get("trajectory") or []
    if frame_idx + 1 >= len(traj):
        return {}
    dt = (run or {}).get("fixed_delta_seconds") or 0.1
    a0 = traj[frame_idx].get("agents") or {}
    a1 = traj[frame_idx + 1].get("agents") or {}
    out = {}
    for tid, p0 in a0.items():
        p1 = a1.get(tid)
        if not isinstance(p0, dict) or not isinstance(p1, dict):
            continue
        d = math.hypot(p1.get("x", 0.0) - p0.get("x", 0.0),
                       p1.get("y", 0.0) - p0.get("y", 0.0))
        out[str(tid)] = d / dt
    return out


def eligible(speeds):
    return {t: s for t, s in speeds.items()
            if t not in ("-1", "-100") and s >= MIN_SPEED}


def main():
    if not NEW.exists():
        print(f"[abort] {NEW} not found")
        return

    scen = sorted({p.name.split("__")[0] for p in GOOD.glob("*__clean.json")})
    attacks = ["sensor_spoofing", "fake_emergency", "fake_safety",
               "traffic_light_tampering", "universal_perturbation", "sybil"]

    print("WHICH ATTACKS FOUND TARGETS IN EACH BATCH")
    print("=" * 94)
    print(f"{'scenario':<42}{'attack':<26}{'good batch':<14}{'new batch'}")
    print("-" * 94)
    regressed = []
    for s in scen:
        for a in attacks:
            g = targets(record(GOOD / f"{s}__{a}__attacked.json"))
            n = targets(record(NEW / f"{s}__{a}__attacked.json"))
            if g == n or (g is None and n is None):
                continue
            gs = ",".join(g) if g else "none"
            ns = ",".join(n) if n else "none"
            flag = ""
            if g and not n:
                regressed.append((s, a, g))
                flag = "  <-- lost"
            print(f"{s[:40]:<42}{a:<26}{gs:<14}{ns}{flag}")
    print("-" * 94)
    print(f"  {len(regressed)} attack/scenario pairs lost their targets\n")

    if not regressed:
        print("  nothing regressed; the batches agree")
        return

    print("WHAT THE TWO CLEAN REPLAYS LOOK LIKE AT THE INJECTION FRAME")
    print("=" * 94)
    print("If the recorded scene itself differs between batches, the target")
    print("selector is behaving correctly and the simulation is what changed.")
    print()
    for s in sorted({r[0] for r in regressed}):
        gc = record(GOOD / f"{s}__clean.json")
        nc = record(NEW / f"{s}__clean.json")
        start = (gc or {}).get("attack_start_frame", 10)
        gsp, nsp = speeds_at_frame(gc, start), speeds_at_frame(nc, start)
        ge, ne = eligible(gsp), eligible(nsp)
        print(f"  {s}")
        print(f"     good batch: {len(gsp):>3} agents, {len(ge):>3} above "
              f"{MIN_SPEED} m/s")
        if ge:
            top = sorted(ge.items(), key=lambda kv: -kv[1])[:4]
            print(f"                 fastest {', '.join(f'{k}={v:.1f}' for k, v in top)}")
        print(f"     new batch : {len(nsp):>3} agents, {len(ne):>3} above "
              f"{MIN_SPEED} m/s")
        if ne:
            top = sorted(ne.items(), key=lambda kv: -kv[1])[:4]
            print(f"                 fastest {', '.join(f'{k}={v:.1f}' for k, v in top)}")
        if not gsp or not nsp:
            print("     (could not read speeds from one of the replays)")
        print()

    print("FRAME COUNTS AND METADATA")
    print("=" * 94)
    print(f"{'scenario':<42}{'good frames':<14}{'new frames':<14}{'start frame'}")
    print("-" * 94)
    for s in scen:
        gc = record(GOOD / f"{s}__clean.json")
        nc = record(NEW / f"{s}__clean.json")
        gf = len((gc or {}).get("trajectory") or [])
        nf = len((nc or {}).get("trajectory") or [])
        st = (gc or {}).get("attack_start_frame", "?")
        mark = "  <-- differs" if gf != nf else ""
        print(f"{s[:40]:<42}{gf:<14}{nf:<14}{st}{mark}")
    print("-" * 94)

    print()
    print("READING")
    print("=" * 94)
    print("  If the two clean replays show the same agents at the same speeds,")
    print("  the regression is in the attack code or its invocation, and a")
    print("  rerun on the current code should recover the targets.")
    print()
    print("  If the new batch's clean replay has fewer moving agents, the")
    print("  simulation itself differed -- a different CARLA version, map")
    print("  package, or traffic seed -- and the selector was right to refuse.")
    print("  In that case the rerun needs the original conditions, not just")
    print("  the original command.")


if __name__ == "__main__":
    main()