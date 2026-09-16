import json
import math
import sys
from itertools import combinations
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
SEEDS = [11, 22, 33, 44, 55]
SPAWNED_KEYS = ("obstacle_actor_ids", "emergency_actor_ids", "sybil_actor_ids")


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


def spawned_ids(record):
    if record is None:
        return []
    md = getattr(record, "metadata", {}) or {}
    out = []
    for k in SPAWNED_KEYS:
        out.extend(md.get(k, []) or [])
    return out


def snapshot(replay_state, world, extra_ids):
    agents, spawned = {}, {}
    for tid, agent in replay_state.agents.items():
        if agent.actor is None:
            continue
        try:
            if agent.actor.is_alive:
                t = agent.actor.get_transform()
                agents[str(tid)] = (t.location.x, t.location.y)
        except RuntimeError:
            continue
    for aid in extra_ids:
        try:
            a = world.get_actor(aid)
            if a is not None and a.is_alive:
                t = a.get_transform()
                spawned[str(aid)] = (t.location.x, t.location.y)
        except RuntimeError:
            continue
    return {"agents": agents, "spawned": spawned}


def run(client, scenario_dir, scenario, tm, release_ids, seed,
        attack_fn=None, attack_kwargs=None, n_frames=60):
    replay_state, _ = load_scenario_world(client, scenario_dir, scenario)
    world = replay_state.world

    set_sync(world, tm, True)
    tm.set_random_device_seed(seed)
    world.tick()
    spawn_agents_at_frame(replay_state, frame_idx=0)
    world.tick()

    total = min(n_frames, len(replay_state.frame_files))
    released = set(release_ids or [])
    record, extra = None, []
    traj = []

    for f in range(total):
        if f == ATTACK_START_FRAME:
            if attack_fn is not None:
                kw = dict(attack_kwargs or {})
                kw["start_frame_idx"] = f
                record = attack_fn(replay_state, **kw)
                extra = spawned_ids(record)
            release(replay_state, released, tm)

        skip = released if f >= ATTACK_START_FRAME else set()
        apply_frame_state(replay_state, f, skip_track_ids=skip)
        world.tick()
        traj.append(snapshot(replay_state, world, extra))

    if record is not None:
        cleanup_attack_record(replay_state, record)
    cleanup_replay(replay_state)
    set_sync(world, tm, False)
    return {"traj": traj, "spawned_ids": extra}


def divergence(a, b, released):
    d = []
    for fa, fb in zip(a["traj"], b["traj"]):
        for tid in released:
            k = str(tid)
            if k in fa["agents"] and k in fb["agents"]:
                d.append(math.dist(fa["agents"][k], fb["agents"][k]))
    arr = np.array(d) if d else np.array([0.0])
    return float(arr.mean()), float(arr.max())


def closest_approach(run_result, released):
    out = {}
    for tid in released:
        k = str(tid)
        best = math.inf
        for fr in run_result["traj"]:
            if k not in fr["agents"]:
                continue
            for pos in fr["spawned"].values():
                best = min(best, math.dist(fr["agents"][k], pos))
        out[k] = None if best is math.inf else best
    return out


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

    print(f"scenario : {scenario}")
    print(f"frames   : {args.frames}")
    print(f"seeds    : {SEEDS}\n")

    probe, _ = load_scenario_world(client, scenario_dir, scenario)
    set_sync(probe.world, tm, True)
    tm.set_random_device_seed(SEEDS[0])
    probe.world.tick()
    spawn_agents_at_frame(probe, frame_idx=0)
    probe.world.tick()
    rec = inject_sensor_spoofing(probe, start_frame_idx=ATTACK_START_FRAME, traffic_manager=tm)
    released = list(getattr(rec, "affected_track_ids", []) or [])
    cleanup_attack_record(probe, rec)
    cleanup_replay(probe)
    set_sync(probe.world, tm, False)

    if not released:
        print("[abort] attack selected no targets in this scenario; nothing to measure.")
        return
    print(f"agents released in every run: {released}\n")

    akw = {"traffic_manager": tm}
    print("running 12 replays...")
    ctl = {}
    att = {}
    for s in SEEDS:
        ctl[s] = run(client, scenario_dir, scenario, tm, released, s, n_frames=args.frames)
        att[s] = run(client, scenario_dir, scenario, tm, released, s,
                     attack_fn=inject_sensor_spoofing, attack_kwargs=akw, n_frames=args.frames)
        print(f"   seed {s} done")
    ctl_rep = run(client, scenario_dir, scenario, tm, released, SEEDS[0], n_frames=args.frames)
    att_rep = run(client, scenario_dir, scenario, tm, released, SEEDS[0],
                  attack_fn=inject_sensor_spoofing, attack_kwargs=akw, n_frames=args.frames)
    print("   repeat runs done\n")

    print("=" * 78)
    print("1. DETERMINISM -- same seed, same condition, run twice")
    print("=" * 78)
    cm, cx = divergence(ctl[SEEDS[0]], ctl_rep, released)
    am, ax = divergence(att[SEEDS[0]], att_rep, released)
    print(f"   control  vs control   mean {cm:.6f}  max {cx:.6f}")
    print(f"   attacked vs attacked  mean {am:.6f}  max {ax:.6f}")
    deterministic = cx < 1e-6 and ax < 1e-6
    print(f"   -> {'deterministic' if deterministic else 'NOT DETERMINISTIC -- results below are unreliable'}\n")

    print("=" * 78)
    print("2. NOISE FLOOR -- control vs control across different seeds (no attack)")
    print("=" * 78)
    noise = []
    for a, b in combinations(SEEDS, 2):
        m, x = divergence(ctl[a], ctl[b], released)
        noise.append(m)
        print(f"   seeds {a:>3} vs {b:>3}   mean {m:8.4f}  max {x:8.4f}")
    noise = np.array(noise)
    print(f"   -> noise floor: mean {noise.mean():.4f}, std {noise.std():.4f}, "
          f"range {noise.min():.4f}-{noise.max():.4f}\n")

    print("=" * 78)
    print("3. EFFECT -- attacked vs control at the SAME seed")
    print("=" * 78)
    effect = []
    for s in SEEDS:
        m, x = divergence(att[s], ctl[s], released)
        effect.append(m)
        print(f"   seed {s:>3}          mean {m:8.4f}  max {x:8.4f}")
    effect = np.array(effect)
    print(f"   -> effect: mean {effect.mean():.4f}, std {effect.std():.4f}, "
          f"range {effect.min():.4f}-{effect.max():.4f}\n")

    print("=" * 78)
    print("4. PHYSICAL CHECK -- how close do agents get to the spawned actors?")
    print("=" * 78)
    approaches = {}
    for s in SEEDS[:2]:
        ca = closest_approach(att[s], released)
        approaches[str(s)] = ca
        n_spawned = len(att[s]["spawned_ids"])
        print(f"   seed {s}: {n_spawned} spawned actor(s)")
        for tid, dist in ca.items():
            if dist is None:
                print(f"      agent {tid}: no spawned actor was ever observed")
            else:
                note = "  <- close enough to react" if dist < 15 else "  <- never came near"
                print(f"      agent {tid}: closest approach {dist:7.2f} m{note}")
    print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not deterministic:
        print("Runs are not reproducible even with the seed pinned. Fix that before")
        print("interpreting anything else.")
        verdict = "not_deterministic"
    else:
        overlap = effect.mean() <= noise.mean() + noise.std()
        if noise.max() < 1e-6:
            print("The noise floor is zero: seed choice alone produces no divergence.")
            if effect.mean() > 1e-6:
                print(f"The attack therefore produces a genuine effect (mean {effect.mean():.4f} m).")
                verdict = "real_effect"
            else:
                print("And the attack produces none either -- it has no observable influence.")
                verdict = "no_effect"
        elif overlap:
            print(f"Effect ({effect.mean():.4f} m) lies within the noise floor "
                  f"({noise.mean():.4f} +/- {noise.std():.4f} m).")
            print("The attack is NOT distinguishable from simply changing the random seed.")
            print("Detection metrics built on this comparison would be measuring noise.")
            verdict = "indistinguishable_from_noise"
        else:
            ratio = effect.mean() / noise.mean() if noise.mean() > 0 else float("inf")
            print(f"Effect ({effect.mean():.4f} m) exceeds the noise floor "
                  f"({noise.mean():.4f} +/- {noise.std():.4f} m) by {ratio:.1f}x.")
            print("The attack produces a measurable effect beyond seed variation.")
            verdict = "real_effect"

        near = [d for ca in approaches.values() for d in ca.values() if d is not None and d < 15]
        if not near:
            print()
            print("However, no released agent came within 15 m of a spawned actor. Any")
            print("trajectory difference therefore cannot be a physical reaction to one,")
            print("which argues against the effect being causal regardless of magnitude.")

    out = REPO_ROOT / "outputs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "attack_effect_diagnostic.json", "w") as f:
        json.dump({
            "scenario": scenario, "frames": args.frames, "seeds": SEEDS,
            "released": released,
            "deterministic": bool(deterministic),
            "determinism_control_max": cx, "determinism_attacked_max": ax,
            "noise_floor": {"values": noise.tolist(), "mean": float(noise.mean()),
                            "std": float(noise.std())},
            "effect": {"values": effect.tolist(), "mean": float(effect.mean()),
                       "std": float(effect.std())},
            "closest_approach": approaches,
            "verdict": verdict,
        }, f, indent=2)
    print(f"\n[done] saved to outputs/results/attack_effect_diagnostic.json")


if __name__ == "__main__":
    main()