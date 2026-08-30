import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord

DEFAULT_DURATION_FRAMES = 30
DEFAULT_RATIO = 1.0


def inject_traffic_light_tampering(
    replay_state,
    start_frame_idx: int,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    ratio: float = DEFAULT_RATIO,
) -> AttackRecord:
    import carla

    all_lights = replay_state.world.get_actors().filter("traffic.traffic_light")
    if len(all_lights) == 0:
        return AttackRecord(
            attack_type="traffic_light_tampering",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: no traffic lights found in this town",
            physical_inconsistency="none (no traffic lights present)",
            metadata={"failed": True},
        )

    num_target = max(1, int(len(all_lights) * ratio))
    target_lights = random.sample(list(all_lights), num_target)

    original_states = {}
    for light in target_lights:
        original_states[light.id] = light.get_state()
        light.set_state(carla.TrafficLightState.Red)
        light.freeze(True)

    end_frame = min(start_frame_idx + duration_frames, len(replay_state.frame_files) - 1)

    description = (
        f"{num_target} of {len(all_lights)} traffic lights (ratio={ratio}) were forced to Red and "
        f"frozen starting at frame {start_frame_idx}, regardless of their true signal timing. "
        f"Vehicles approaching these lights may brake or stop for a light state that does not "
        f"reflect the actual intersection's real signal cycle."
    )

    physical_inconsistency = (
        "A vehicle's braking/stopping behavior near a tampered light will appear physically "
        "correct in isolation (real braking, real deceleration), but is a reaction to a "
        "falsified signal state rather than the true traffic light phase -- the inconsistency "
        "is only visible by comparing the vehicle's behavior against the light's original, "
        "untampered state."
    )

    return AttackRecord(
        attack_type="traffic_light_tampering",
        affected_track_ids=[],
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "ratio": ratio,
            "target_light_ids": [light.id for light in target_lights],
            "original_states": {lid: str(state) for lid, state in original_states.items()},
        },
    )


def restore_traffic_lights(replay_state, record: AttackRecord):
    import carla

    original_states = record.metadata.get("original_states", {})
    all_lights = {light.id: light for light in replay_state.world.get_actors().filter("traffic.traffic_light")}

    for light_id_str, state_str in original_states.items():
        light_id = int(light_id_str) if isinstance(light_id_str, str) else light_id_str
        light = all_lights.get(light_id)
        if light is not None:
            light.freeze(False)