import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord
from src.attacks.environment_attacks.attack_common import (
    select_unique_lane_targets,
    spawn_static_obstacle,
    offset_location_along_heading,
    enforced_behaviour,
)

DEFAULT_DURATION_FRAMES = 30
DEFAULT_COUNT = 1
OBSTACLE_OFFSET_METERS = 15.0


def inject_fake_safety(
    replay_state,
    start_frame_idx: int,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    count: int = DEFAULT_COUNT,
) -> AttackRecord:
    import math

    targets = select_unique_lane_targets(replay_state, count=count)

    if not targets:
        return AttackRecord(
            attack_type="fake_safety",
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

        affected_track_ids.append(track_id)
        if obstacle_actor is not None:
            obstacle_actors.append(obstacle_actor.id)

    end_frame = min(start_frame_idx + duration_frames, len(replay_state.frame_files) - 1)

    description = (
        f"{len(obstacle_actors)} fake safety-hazard obstacles were spawned {OBSTACLE_OFFSET_METERS}m "
        f"ahead of {len(affected_track_ids)} target agent(s) (track_ids={affected_track_ids}) at frame "
        f"{start_frame_idx}, broadcasting a false road-hazard condition that does not exist."
    )

    physical_inconsistency = (
        "The obstacle has no corresponding real-world cause (no debris source, no stopped vehicle "
        "with a plausible breakdown) and no prior existence in the scene before this frame -- it is "
        "a fabricated safety event with no physical origin."
    )

    return AttackRecord(
        attack_type="fake_safety",
        affected_track_ids=affected_track_ids,
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "enforced_behaviour": enforced_behaviour(
                affected_track_ids, 0.5,
                "target agents slow to 0.5 m/s for the fabricated hazard, "
                "reproducing the slowDown(vehicle, 0.5, 20) override of the "
                "original SUMO attack"),
            "count": count,
            "obstacle_actor_ids": obstacle_actors,
            "obstacle_offset_m": OBSTACLE_OFFSET_METERS,
        },
    )