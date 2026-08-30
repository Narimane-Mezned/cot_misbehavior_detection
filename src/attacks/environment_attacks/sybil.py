import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord

DEFAULT_DURATION_FRAMES = 30
DEFAULT_COUNT = 5
CRAWL_SPEED_MIN = 1.0
CRAWL_SPEED_MAX = 3.0
SPAWN_JITTER_METERS = 30.0


def inject_sybil(
    replay_state,
    start_frame_idx: int,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    count: int = DEFAULT_COUNT,
    attacker_track_id: int = None,
) -> AttackRecord:
    import carla

    real_track_ids = [tid for tid, agent in replay_state.agents.items() if agent.actor is not None]
    if not real_track_ids:
        return AttackRecord(
            attack_type="sybil",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: no real agents present to clone identity from",
            physical_inconsistency="none (no agents present)",
            metadata={"failed": True},
        )

    if attacker_track_id is None or attacker_track_id not in real_track_ids:
        attacker_track_id = real_track_ids[0]

    attacker_agent = replay_state.agents[attacker_track_id]
    attacker_transform = attacker_agent.actor.get_transform()
    blueprint_id = attacker_agent.actor.type_id

    world = replay_state.world
    blueprint_library = world.get_blueprint_library()
    bp_matches = blueprint_library.filter(blueprint_id)
    bp = bp_matches[0] if len(bp_matches) > 0 else blueprint_library.filter("vehicle.*")[0]

    created_sybil_ids = []

    for i in range(count):
        jitter_x = random.uniform(-SPAWN_JITTER_METERS, SPAWN_JITTER_METERS)
        jitter_y = random.uniform(-SPAWN_JITTER_METERS, SPAWN_JITTER_METERS)
        spawn_location = carla.Location(
            x=attacker_transform.location.x + jitter_x,
            y=attacker_transform.location.y + jitter_y,
            z=attacker_transform.location.z,
        )
        transform = carla.Transform(spawn_location, attacker_transform.rotation)

        actor = world.try_spawn_actor(bp, transform)
        if actor is None:
            continue

        actor.set_target_velocity(carla.Vector3D(x=random.uniform(CRAWL_SPEED_MIN, CRAWL_SPEED_MAX)))

        created_sybil_ids.append(actor.id)

    if not created_sybil_ids:
        return AttackRecord(
            attack_type="sybil",
            affected_track_ids=[attacker_track_id],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: could not spawn any Sybil clones",
            physical_inconsistency="none (spawn failed)",
            metadata={"failed": True},
        )

    end_frame = min(start_frame_idx + duration_frames, len(replay_state.frame_files) - 1)

    description = (
        f"Vehicle track_id={attacker_track_id}'s identity (same vehicle type, spawned near its position) "
        f"was cloned into {len(created_sybil_ids)} fake Sybil vehicles at frame {start_frame_idx}, each "
        f"moving at crawling speed ({CRAWL_SPEED_MIN}-{CRAWL_SPEED_MAX} m/s), lane changes disabled. "
        f"All {len(created_sybil_ids)} clones claim to be legitimate distinct vehicles."
    )

    physical_inconsistency = (
        f"{len(created_sybil_ids)} vehicles of identical type appeared simultaneously within "
        f"{SPAWN_JITTER_METERS}m of each other at the same frame, each with no independent prior "
        f"trajectory -- multiple genuinely distinct vehicles would not spawn into existence "
        f"simultaneously in tight spatial clustering."
    )

    return AttackRecord(
        attack_type="sybil",
        affected_track_ids=[attacker_track_id],
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "attacker_track_id": attacker_track_id,
            "sybil_actor_ids": created_sybil_ids,
            "count": count,
        },
    )