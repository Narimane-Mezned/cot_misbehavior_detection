import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.carla_replay import (
    connect_carla, load_scenario_world, spawn_agents_at_frame, apply_frame_state,
    list_scenarios, cleanup_replay, cleanup_attack_record,
)

ATTACK_START_FRAME = 10
FIXED_DELTA_SECONDS = 0.1
TM_SEED = 20260914


def set_sync(world, tm, enabled):
    s = world.get_settings()
    s.synchronous_mode = enabled
    s.fixed_delta_seconds = FIXED_DELTA_SECONDS if enabled else None
    world.apply_settings(s)
    if tm is not None:
        tm.set_synchronous_mode(enabled)


def release(replay_state, track_ids, tm):
    from src.attacks.environment_attacks.attack_common import release_to_autopilot
    for tid in track_ids:
        agent = replay_state.agents.get(tid)
        if agent is None or agent.actor is None:
            continue
        try:
            if agent.actor.is_alive and agent.actor.type_id.startswith("vehicle."):
                release_to_autopilot(replay_state, tid, tm)
        except RuntimeError:
            continue


def record(replay_state, frame_idx):
    entry = {}
    for tid, agent in replay_state.agents.items():
        if agent.actor is None:
            continue
        try:
            if not agent.actor.is_alive:
                continue
            tf = agent.actor.get_transform()
            entry[str(tid)] = (tf.location.x, tf.location.y)
        except RuntimeError:
            continue
    return entry


def run(client, scenario_dir, scenario_name, tm, release_ids,
        attack_fn=None, attack_kwargs=None, seed=None, n_frames=60):
    replay_state, _ = load_scenario_world(client, scenario_dir, scenario_name)
    world = replay_state.world

    set_sync(world, tm, True)
    if seed is not None:
        tm.set_random_device_seed(seed)
    world.tick()
    spawn_agents_at_frame(replay_state, frame_idx=0)
    world.tick()

    total = min(n_frames, len(replay_state.frame_files))
    released = set(release_ids or [])
    attack_record = None
    traj = []

    for f in range(total):
        if f == ATTACK_START_FRAME:
            if attack_fn is not None:
                kw = dict(attack_kwargs or {})
                kw["start_frame_idx"] = f
                attack_record = attack_fn(replay_state, **kw)
            release(replay_state, released, tm)

        skip = released if f >= ATTACK_START_FRAME else set()
        apply_frame_state(replay_state, f, skip_track_ids=skip)
        world.tick()
        traj.append(record(replay_state, f))

    if attack_record is not None:
        cleanup_attack_record(replay_state, attack_record)
    cleanup_replay(replay_state)
    set_sync(world, tm, False)
    return traj


def compare(a, b, released, label):
    divs = []
    for fa, fb in zip(a, b):
        for tid in released:
            k = str(tid)
            if k in fa and k in fb:
                divs.append(math.dist(fa[k], fb[k]))
    d = np.array(divs) if divs else np.array([0.0])
    verdict = "IDENTICAL" if d.max() < 1e-6 else "DIVERGES"
    print(f"  {label:<44}{verdict:<12}mean {d.mean():.4f}  max {d.max():.4f}")
    return bool(d.max() >= 1e-6)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--scenario_type", default="type1_subtype1_normal")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    from src.attacks.environment_attacks.sensor_spoofing import inject_sensor_spoofing

    client = connect_carla(host=args.host, port=args.port, timeout=args.timeout)
    tm = client.get_trafficmanager()

    scenario_dir = REPO_ROOT / args.data_root / args.scenario_type
    scenario = list_scenarios(scenario_dir)[0]
    print(f"scenario: {scenario}   frames: {args.frames}\n")

   
    probe = load_scenario_world(client, scenario_dir, scenario)[0]
    set_sync(probe.world, tm, True)
    probe.world.tick()
    spawn_agents_at_frame(probe, frame_idx=0)
    probe.world.tick()
    rec = inject_sensor_spoofing(probe, start_frame_idx=ATTACK_START_FRAME, traffic_manager=tm)
    released = list(getattr(rec, "affected_track_ids", []) or [])
    cleanup_attack_record(probe, rec)
    cleanup_replay(probe)
    set_sync(probe.world, tm, False)
    print(f"agents released in every run below: {released}\n")

    if not released:
        print("[abort] the attack selected no targets in this scenario; "
              "nothing to compare. Try a different scenario_type.")
        return

    print("=" * 78)
    print("TEST A -- control vs control, no seed pinned, no attack")
    print("=" * 78)
    a1 = run(client, scenario_dir, scenario, tm, released, n_frames=args.frames)
    a2 = run(client, scenario_dir, scenario, tm, released, n_frames=args.frames)
    a_div = compare(a1, a2, released, "two identical control runs")

    print()
    print("=" * 78)
    print(f"TEST B -- control vs control, Traffic Manager seed pinned to {TM_SEED}")
    print("=" * 78)
    b1 = run(client, scenario_dir, scenario, tm, released, seed=TM_SEED, n_frames=args.frames)
    b2 = run(client, scenario_dir, scenario, tm, released, seed=TM_SEED, n_frames=args.frames)
    b_div = compare(b1, b2, released, "two identical control runs, seeded")

    print()
    print("=" * 78)
    print("TEST C -- attacked vs control, seed pinned")
    print("=" * 78)
    c_att = run(client, scenario_dir, scenario, tm, released,
                attack_fn=inject_sensor_spoofing,
                attack_kwargs={"traffic_manager": tm}, seed=TM_SEED, n_frames=args.frames)
    c_ctl = run(client, scenario_dir, scenario, tm, released, seed=TM_SEED, n_frames=args.frames)
    c_div = compare(c_att, c_ctl, released, "attacked vs control, seeded")

    print()
    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    if a_div and not b_div:
        print("Traffic Manager nondeterminism confirmed, and pinning the seed removes it.")
        print("Fix: set traffic_manager.set_random_device_seed() in the recording harness.")
        if c_div:
            print("With the seed pinned, the attack produces a genuine, measurable effect.")
        else:
            print("With the seed pinned, the attack produces NO measurable effect. The")
            print("earlier divergence was entirely nondeterminism; the attack itself does")
            print("nothing observable here, which needs addressing separately.")
    elif not a_div:
        print("Two identical control runs are byte-identical, so the simulator is")
        print("deterministic and nondeterminism is NOT the cause. The attacked-vs-control")
        print("difference must come from calling attack_fn. Further instrumentation of")
        print("the harness is required.")
        if c_div:
            print("Test C confirms the attacked run differs -- the cause is inside attack_fn.")
    else:
        print("Seeding did not remove the divergence. The simulator is nondeterministic in")
        print("a way the Traffic Manager seed does not control; this needs investigating")
        print("before any detection evaluation can be trusted.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "determinism_diagnostic.json", "w") as f:
        json.dump({"scenario": scenario, "released": released, "frames": args.frames,
                   "tm_seed": TM_SEED,
                   "A_control_vs_control_diverges": bool(a_div),
                   "B_seeded_control_vs_control_diverges": bool(b_div),
                   "C_seeded_attacked_vs_control_diverges": bool(c_div)}, f, indent=2)
    print(f"\n[done] saved to outputs/results/determinism_diagnostic.json")


if __name__ == "__main__":
    main()