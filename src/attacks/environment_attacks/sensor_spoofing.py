import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord
from src.attacks.environment_attacks.attack_common import (
    select_unique_lane_targets,
    spawn_static_obstacle,
    offset_location_along_heading,
    release_to_autopilot,
)

DEFAULT_DURATION_FRAMES = 30
DEFAULT_NUM_OBSTACLES = 2
OBSTACLE_OFFSET_METERS = 25.0


def inject_sensor_spoofing(
    replay_state,
    start_frame_idx: int,
    traffic_manager,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    num_obstacles: int = DEFAULT_NUM_OBSTACLES,
) -> AttackRecord:
    import math

    targets = select_unique_lane_targets(replay_state, count=num_obstacles, sort_by_speed_desc=True)

    if not targets:
        return AttackRecord(
            attack_type="sensor_spoofing",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: no eligible target agents found",
            physical_inconsistency="none (no targets available)",
            metadata={"failed": True},
        )

    obstacle_actors = []
    affected_track_ids = []

    for track_id, agent, lane_id, speed in targets:
        transform = agent.actor.get_transform()
        yaw_radians = math.radians(transform.rotation.yaw)
        obstacle_location = offset_location_along_heading(
            transform.location, yaw_radians, OBSTACLE_OFFSET_METERS
        )
        obstacle_actor = spawn_static_obstacle(replay_state, obstacle_location)

        if obstacle_actor is not None:
            obstacle_actors.append(obstacle_actor.id)
            affected_track_ids.append(track_id)
            release_to_autopilot(replay_state, track_id, traffic_manager)

    if not obstacle_actors:
        return AttackRecord(
            attack_type="sensor_spoofing",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: could not spawn any obstacles",
            physical_inconsistency="none (spawn failed)",
            metadata={"failed": True},
        )

    end_frame = min(start_frame_idx + duration_frames, len(replay_state.frame_files) - 1)

    description = (
        f"{len(obstacle_actors)} static, physics-disabled obstacles were spawned {OBSTACLE_OFFSET_METERS}m "
        f"ahead of {len(affected_track_ids)} fast-moving target agents (track_ids={affected_track_ids}) "
        f"at frame {start_frame_idx}. Target agents were released to CARLA's own Traffic Manager for the "
        f"attack duration so they detect and react to the obstacles via their own sensors, not a scripted "
        f"override -- matching the original attack's reliance on the simulator's own reactive behavior."
    )

    physical_inconsistency = (
        "The obstacle actors have no prior trajectory history and no physics simulation -- they did not "
        "arrive by driving, they simply appeared. A real static hazard would typically correspond to a "
        "stopped vehicle or debris with its own plausible arrival history, which these lack."
    )

    return AttackRecord(
        attack_type="sensor_spoofing",
        affected_track_ids=affected_track_ids,
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "num_obstacles": num_obstacles,
            "obstacle_actor_ids": obstacle_actors,
            "obstacle_offset_m": OBSTACLE_OFFSET_METERS,
        },
    )