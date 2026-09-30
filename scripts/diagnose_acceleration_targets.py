import json, glob, math
from pathlib import Path

ATTACKS = ["sensor_spoofing", "forced_acceleration"]

def speeds_at(run, frame_idx):
    traj = run.get("trajectory") or []
    if frame_idx >= len(traj):
        return {}
    out = {}
    for a in traj[frame_idx].get("agents", []):
        v = a.get("velocity") or {}
        sp = math.hypot(v.get("x", 0.0), v.get("y", 0.0))
        out[str(a.get("track_id"))] = sp
    return out

for scen in sorted({Path(f).name.split("__")[0]
                    for f in glob.glob("data/attack_trajectories/*__clean.json")}):
    clean = json.load(open(f"data/attack_trajectories/{scen}__clean.json"))
    print("=" * 78)
    print(scen)
    for atk in ATTACKS:
        p = f"data/attack_trajectories/{scen}__{atk}__attacked.json"
        try:
            r = json.load(open(p))
        except FileNotFoundError:
            continue
        rec = r.get("attack_record") or {}
        bh = (rec.get("metadata") or {}).get("enforced_behaviour") or {}
        start = r.get("attack_start_frame", rec.get("start_frame"))
        ids = bh.get("track_ids")
        print(f"  {atk:<22} start_frame={start}  targets={ids if ids else 'NONE'}")

    start = clean.get("attack_start_frame") or 10
    sp = speeds_at(clean, start)
    movers = {k: v for k, v in sp.items() if v >= 1.0 and k not in ("-1", "-100")}
    print(f"  clean replay at frame {start}: {len(sp)} agents, "
          f"{len(movers)} above 1.0 m/s")
    if movers:
        top = sorted(movers.items(), key=lambda kv: -kv[1])[:5]
        print("    fastest:", ", ".join(f"{k}={v:.2f}" for k, v in top))