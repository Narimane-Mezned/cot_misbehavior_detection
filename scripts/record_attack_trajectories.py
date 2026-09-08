import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.carla_replay import (
    connect_carla, load_scenario_world, spawn_agents_at_frame, apply_frame_state,
    list_scenarios, cleanup_replay, cleanup_attack_record,
)

ATTACK_START_FRAME = 10
ATTACK_DURATION_FRAMES = 30
FIXED_DELTA_SECONDS = 0.05

SPAWNED_ACTOR_KEYS = ("obstacle_actor_ids", "emergency_actor_ids", "sybil_actor_ids")


def set_synchronous_mode(world, traffic_manager, enabled: bool):
    settings = world.get_settings()
    settings.synchronous_mode = enabled
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS if enabled else None
    world.apply_settings(settings)
    if traffic_manager is not None:
        traffic_manager.set_synchronous_mode(enabled)


def collect_spawned_actor_ids(attack_record) -> list:
    if attack_record is None:
        return []
    metadata = getattr(attack_record, "metadata", {}) or {}
    ids = []
    for key in SPAWNED_ACTOR_KEYS:
        ids.extend(metadata.get(key, []) or [])
    return ids


def record_frame(world, replay_state, frame_idx: int, spawned_actor_ids: list) -> dict:
    entry = {"frame_idx": frame_idx, "agents": {}, "spawned": {}}

    for track_id, agent in replay_state.agents.items():
        if agent.actor is None:
            continue
        try:
            if not agent.actor.is_alive:
                continue
            tf = agent.actor.get_transform()
            entry["agents"][str(track_id)] = {
                "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                "yaw_deg": tf.rotation.yaw,
            }
        except RuntimeError:
            continue

    for actor_id in spawned_actor_ids:
        try:
            actor = world.get_actor(actor_id)
            if actor is None or not actor.is_alive:
                continue
            tf = actor.get_transform()
            entry["spawned"][str(actor_id)] = {
                "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                "yaw_deg": tf.rotation.yaw,
            }
        except RuntimeError:
            continue

    return entry


def apply_perturbation_offset(replay_state, perturbation: dict):
    import carla
    dx, dy = perturbation["position"]
    dheading_deg = math.degrees(perturbation["heading"])

    for agent in replay_state.agents.values():
        if agent.actor is None:
            continue
        try:
            if not agent.actor.is_alive:
                continue
            tf = agent.actor.get_transform()
            new_tf = carla.Transform(
                carla.Location(x=tf.location.x + dx, y=tf.location.y + dy, z=tf.location.z),
                carla.Rotation(pitch=tf.rotation.pitch,
                               yaw=tf.rotation.yaw + dheading_deg,
                               roll=tf.rotation.roll),
            )
            agent.actor.set_transform(new_tf)
        except RuntimeError:
            continue


def release_agents_to_traffic_manager(replay_state, track_ids, traffic_manager):
    for track_id in track_ids:
        agent = replay_state.agents.get(track_id)
        if agent is None or agent.actor is None:
            continue
        try:
            if agent.actor.is_alive and agent.actor.type_id.startswith("vehicle."):
                agent.actor.set_autopilot(True, traffic_manager.get_port())
        except RuntimeError:
            continue


def run_replay(client, scenario_type_dir, scenario_name, traffic_manager, mode,
               attack_fn=None, attack_kwargs=None, release_track_ids=None,
               release_all=False, max_frames=None):
    replay_state, meta = load_scenario_world(client, scenario_type_dir, scenario_name)
    world = replay_state.world

    set_synchronous_mode(world, traffic_manager, True)
    world.tick()
    spawn_agents_at_frame(replay_state, frame_idx=0)
    world.tick()

    n_frames = len(replay_state.frame_files)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    attack_record = None
    spawned_actor_ids = []
    released = set(release_track_ids or [])
    perturbation = None
    trajectory = []

    for frame_idx in range(n_frames):
        if frame_idx == ATTACK_START_FRAME:
            if mode == "attacked" and attack_fn is not None:
                kwargs = dict(attack_kwargs or {})
                kwargs["start_frame_idx"] = frame_idx
                attack_record = attack_fn(replay_state, **kwargs)
                spawned_actor_ids = collect_spawned_actor_ids(attack_record)

                affected = set(getattr(attack_record, "affected_track_ids", []) or [])
                if release_all:
                    affected = {tid for tid, a in replay_state.agents.items() if a.actor is not None}
                released = affected

                metadata = getattr(attack_record, "metadata", {}) or {}
                perturbation = metadata.get("perturbation")

                release_agents_to_traffic_manager(replay_state, released, traffic_manager)
                print(f"    [attack] injected at frame {frame_idx}; "
                      f"released {len(released)} agent(s); "
                      f"spawned {len(spawned_actor_ids)} actor(s)")

            elif mode == "control" and released:
                release_agents_to_traffic_manager(replay_state, released, traffic_manager)
                print(f"    [control] released the same {len(released)} agent(s), no attack")

        attack_window_active = ATTACK_START_FRAME <= frame_idx < ATTACK_START_FRAME + ATTACK_DURATION_FRAMES
        skip = released if attack_window_active else set()

        apply_frame_state(replay_state, frame_idx, skip_track_ids=skip)

        if mode == "attacked" and perturbation is not None and attack_window_active:
            apply_perturbation_offset(replay_state, perturbation)

        world.tick()
        trajectory.append(record_frame(world, replay_state, frame_idx, spawned_actor_ids))

    if attack_record is not None:
        cleanup_attack_record(replay_state, attack_record)
    cleanup_replay(replay_state)
    set_synchronous_mode(world, traffic_manager, False)

    return {
        "trajectory": trajectory,
        "attack_record": attack_record,
        "meta": meta,
        "released_track_ids": sorted(released),
        "spawned_actor_ids": spawned_actor_ids,
    }


def serialize_attack_record(record) -> dict:
    if record is None:
        return None
    metadata = {}
    for k, v in (record.metadata or {}).items():
        metadata[k] = list(v) if isinstance(v, tuple) else v
    return {
        "attack_type": record.attack_type,
        "affected_track_ids": list(record.affected_track_ids or []),
        "start_frame": record.start_frame,
        "end_frame": record.end_frame,
        "description": record.description,
        "physical_inconsistency": record.physical_inconsistency,
        "metadata": metadata,
    }


def save_run(output_dir: Path, scenario_name: str, run_label: str, attack_type, result: dict):
    payload = {
        "scenario_name": scenario_name,
        "run_label": run_label,
        "attack_type": attack_type,
        "fixed_delta_seconds": FIXED_DELTA_SECONDS,
        "attack_start_frame": ATTACK_START_FRAME,
        "attack_duration_frames": ATTACK_DURATION_FRAMES,
        "meta": result["meta"],
        "released_track_ids": result["released_track_ids"],
        "spawned_actor_ids": result["spawned_actor_ids"],
        "attack_record": serialize_attack_record(result["attack_record"]),
        "trajectory": result["trajectory"],
        "note": ("Positions are recorded per frame. Velocity must be derived as "
                 "(pos[t] - pos[t-1]) / fixed_delta_seconds -- CARLA's get_velocity() is "
                 "meaningless for teleported replay actors."),
    }
    out = output_dir / f"{scenario_name}__{run_label}.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--scenario_type", type=str, default="type1_subtype1_normal")
    parser.add_argument("--num_scenarios", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--attacks", type=str,
                        default="sensor_spoofing,fake_emergency,fake_safety,traffic_light_tampering,universal_perturbation,sybil")
    args = parser.parse_args()

    from src.attacks.environment_attacks.sensor_spoofing import inject_sensor_spoofing
    from src.attacks.environment_attacks.fake_emergency import inject_fake_emergency
    from src.attacks.environment_attacks.fake_safety import inject_fake_safety
    from src.attacks.environment_attacks.traffic_light_tampering import inject_traffic_light_tampering
    from src.attacks.environment_attacks.universal_perturbation import inject_universal_perturbation
    from src.attacks.environment_attacks.sybil import inject_sybil

    client = connect_carla(host=args.host, port=args.port)
    traffic_manager = client.get_trafficmanager()

    registry = {
        "sensor_spoofing": (inject_sensor_spoofing, {"traffic_manager": traffic_manager}, False),
        "fake_emergency": (inject_fake_emergency, {"traffic_manager": traffic_manager}, False),
        "fake_safety": (inject_fake_safety, {}, True),
        "traffic_light_tampering": (inject_traffic_light_tampering, {}, True),
        "universal_perturbation": (inject_universal_perturbation, {}, False),
        "sybil": (inject_sybil, {}, True),
    }

    requested = [a.strip() for a in args.attacks.split(",") if a.strip()]
    scenario_type_dir = REPO_ROOT / args.data_root / args.scenario_type
    scenario_names = list_scenarios(scenario_type_dir)[:args.num_scenarios]

    output_dir = REPO_ROOT / "data" / "attack_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] {len(scenario_names)} scenario(s) x {len(requested)} attack(s)")
    print(f"[setup] Each attack produces an 'attacked' run AND a matched 'control' run")
    print(f"[setup] (control = same agents released to Traffic Manager, but NO attack)")
    print(f"[setup] Plus one pure 'clean' replay per scenario.")
    print(f"[setup] Output: {output_dir}\n")

    for scenario_name in scenario_names:
        print(f"[scenario] {scenario_name}")

        print("  [clean] pure replay, no release, no attack...")
        try:
            result = run_replay(client, scenario_type_dir, scenario_name, traffic_manager,
                                mode="clean", max_frames=args.max_frames)
            out = save_run(output_dir, scenario_name, "clean", None, result)
            print(f"  [clean] {len(result['trajectory'])} frames -> {out.name}")
        except Exception as e:
            print(f"  [clean] FAILED: {type(e).__name__}: {e}")
            continue

        for attack_name in requested:
            if attack_name not in registry:
                print(f"  [{attack_name}] unknown, skipping")
                continue
            attack_fn, attack_kwargs, release_all = registry[attack_name]

            print(f"  [{attack_name}] attacked run...")
            try:
                attacked = run_replay(client, scenario_type_dir, scenario_name, traffic_manager,
                                      mode="attacked", attack_fn=attack_fn,
                                      attack_kwargs=attack_kwargs, release_all=release_all,
                                      max_frames=args.max_frames)
                out = save_run(output_dir, scenario_name, f"{attack_name}__attacked",
                               attack_name, attacked)
                print(f"  [{attack_name}] {len(attacked['trajectory'])} frames -> {out.name}")
            except Exception as e:
                print(f"  [{attack_name}] attacked run FAILED: {type(e).__name__}: {e}")
                continue

            print(f"  [{attack_name}] matched control run (same agents released, no attack)...")
            try:
                control = run_replay(client, scenario_type_dir, scenario_name, traffic_manager,
                                     mode="control",
                                     release_track_ids=attacked["released_track_ids"],
                                     max_frames=args.max_frames)
                out = save_run(output_dir, scenario_name, f"{attack_name}__control",
                               None, control)
                print(f"  [{attack_name}] {len(control['trajectory'])} frames -> {out.name}")
            except Exception as e:
                print(f"  [{attack_name}] control run FAILED: {type(e).__name__}: {e}")

        print()

    print(f"[done] Saved to {output_dir}")
    print("[done] Please send the whole data/attack_trajectories/ folder back.")


if __name__ == "__main__":
    main()