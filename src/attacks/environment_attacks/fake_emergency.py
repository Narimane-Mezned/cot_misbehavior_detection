import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord
from src.attacks.environment_attacks.attack_common import (
    select_unique_lane_targets,
    offset_location_along_heading,
    release_to_autopilot,
)

DEFAULT_DURATION_FRAMES = 30
DEFAULT_COUNT = 1
DEFAULT_SPEED_MS = 15.0
SPAWN_BEHIND_OFFSET_METERS = 25.0
EMERGENCY_BLUEPRINT_FILTER = "vehicle.dodge.charger_police"


def inject_fake_emergency(
    replay_state,
    start_frame_idx: int,
    traffic_manager,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    count: int = DEFAULT_COUNT,
    speed_ms: float = DEFAULT_SPEED_MS,
) -> AttackRecord:
    import math
    import carla

    targets = select_unique_lane_targets(replay_state, count=count)

    if not targets:
        return AttackRecord(
            attack_type="fake_emergency",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: no eligible target agents found",
            physical_inconsistency="none (no targets available)",
            metadata={"failed": True},
        )

    world = replay_state.world
    blueprint_library = world.get_blueprint_library()
    bp_matches = blueprint_library.filter(EMERGENCY_BLUEPRINT_FILTER)
    bp = bp_matches[0] if len(bp_matches) > 0 else blueprint_library.filter("vehicle.*")[0]

    created_ev_ids = []
    affected_track_ids = []

    for track_id, agent, lane_id, speed in targets:
        transform = agent.actor.get_transform()
        yaw_radians = math.radians(transform.rotation.yaw)
        spawn_location = offset_location_along_heading(
            transform.location, yaw_radians, -SPAWN_BEHIND_OFFSET_METERS
        )
        spawn_transform = carla.Transform(spawn_location, transform.rotation)

        ev_actor = world.try_spawn_actor(bp, spawn_transform)
        if ev_actor is None:
            continue

        try:
            light_state = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2
            ev_actor.set_light_state(carla.VehicleLightState(light_state))
        except Exception:
            pass

        forward_vector = transform.get_forward_vector()
        target_velocity = carla.Vector3D(
            x=forward_vector.x * speed_ms,
            y=forward_vector.y * speed_ms,
            z=0.0,
        )
        ev_actor.set_target_velocity(target_velocity)

        created_ev_ids.append(ev_actor.id)
        affected_track_ids.append(track_id)
        release_to_autopilot(replay_state, track_id, traffic_manager)

    if not created_ev_ids:
        return AttackRecord(
            attack_type="fake_emergency",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: could not spawn any fake emergency vehicles",
            physical_inconsistency="none (spawn failed)",
            metadata={"failed": True},
        )

    end_frame = min(start_frame_idx + duration_frames, len(replay_state.frame_files) - 1)

    description = (
        f"{len(created_ev_ids)} fabricated emergency vehicle(s) were spawned {SPAWN_BEHIND_OFFSET_METERS}m "
        f"behind {len(affected_track_ids)} target agent(s) (track_ids={affected_track_ids}) at frame "
        f"{start_frame_idx}, moving at {speed_ms} m/s with emergency lights active. Target agents were "
        f"released to CARLA's own Traffic Manager for the attack duration so any yielding/reaction "
        f"emerges from the simulator's own collision-avoidance behavior, not a scripted override."
    )

    physical_inconsistency = (
        "The emergency vehicle has no prior trajectory history before this frame -- it did not "
        "gradually enter sensor range, it appeared already at full speed with lights active. A real "
        "emergency vehicle would be observed approaching over several preceding frames."
    )

    return AttackRecord(
        attack_type="fake_emergency",
        affected_track_ids=affected_track_ids,
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "count": count,
            "speed_ms": speed_ms,
            "emergency_actor_ids": created_ev_ids,
            "spawn_behind_offset_m": SPAWN_BEHIND_OFFSET_METERS,
        },
    )