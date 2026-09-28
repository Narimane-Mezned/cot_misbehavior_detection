import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord
from src.attacks.environment_attacks.attack_common import (
    select_unique_lane_targets,
    enforced_behaviour,
)

DEFAULT_DURATION_FRAMES = 30
DEFAULT_TARGET_COUNT = 2
DEFAULT_SPEED_MS = 15.0


def inject_forced_acceleration(
    replay_state,
    start_frame_idx: int,
    traffic_manager,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    target_count: int = DEFAULT_TARGET_COUNT,
    speed_ms: float = DEFAULT_SPEED_MS,
) -> AttackRecord:
    targets = select_unique_lane_targets(replay_state, target_count)

    if not targets:
        return AttackRecord(
            attack_type="forced_acceleration",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: no eligible target agents found",
            physical_inconsistency="none (no targets available)",
            metadata={"failed": True},
        )

    affected_track_ids = [tid for tid, _ in targets]
    end_frame = start_frame_idx + duration_frames

    description = (
        f"A falsified clear-road advisory is broadcast to {len(affected_track_ids)} "
        f"vehicle(s), which respond by accelerating to {speed_ms:.1f} m/s regardless "
        f"of the traffic ahead of them. This is the mirror image of the other six "
        f"attacks in the suite: rather than suppressing motion it induces motion, "
        f"and it exists to test the boundary of a detection measure built around "
        f"vehicles moving slower than predicted."
    )

    physical_inconsistency = (
        "The commanded vehicles accelerate beyond the speed their own recent "
        "trajectory implies, and beyond what the vehicles ahead of them permit. "
        "A vehicle genuinely responding to a clear road would accelerate only as "
        "the gap ahead opened, not before."
    )

    return AttackRecord(
        attack_type="forced_acceleration",
        affected_track_ids=affected_track_ids,
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "enforced_behaviour": enforced_behaviour(
                affected_track_ids, speed_ms,
                f"target agents are forced to {speed_ms} m/s, inverting the "
                f"motion-suppression pattern of the other six attacks"),
            "target_count": target_count,
            "speed_ms": speed_ms,
            "inverts_motion_suppression": True,
        },
    )