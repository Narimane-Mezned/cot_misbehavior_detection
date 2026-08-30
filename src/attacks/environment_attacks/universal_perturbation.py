import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.attacks.environment_attacks.attack_record import AttackRecord

DEFAULT_DURATION_FRAMES = 30
DEFAULT_EPSILON = 0.3
DEFAULT_SCALE_POSITION = 0.5
DEFAULT_SCALE_VELOCITY = 0.3


def generate_perturbation(epsilon: float, scale_position: float, scale_velocity: float, seed: int = None):
    rng = np.random.default_rng(seed)

    raw_pos = np.clip(rng.normal(0, epsilon * scale_position, 2), -epsilon, epsilon)
    raw_vel = np.clip(rng.normal(0, epsilon * scale_velocity, 2), -epsilon * 0.5, epsilon * 0.5)
    raw_head = np.clip(rng.normal(0, epsilon * 0.2, 1)[0], -0.1, 0.1)

    return {
        "position": [float(raw_pos[0]), float(raw_pos[1])],
        "velocity": [float(raw_vel[0]), float(raw_vel[1])],
        "heading": float(raw_head),
    }


def apply_perturbation_to_frame_state(x: float, y: float, yaw_radians: float, perturbation: dict):
    dx, dy = perturbation["position"]
    dheading = perturbation["heading"]
    return x + dx, y + dy, yaw_radians + dheading


def inject_universal_perturbation(
    replay_state,
    start_frame_idx: int,
    duration_frames: int = DEFAULT_DURATION_FRAMES,
    epsilon: float = DEFAULT_EPSILON,
    scale_position: float = DEFAULT_SCALE_POSITION,
    scale_velocity: float = DEFAULT_SCALE_VELOCITY,
    seed: int = None,
) -> AttackRecord:
    affected_track_ids = [tid for tid, agent in replay_state.agents.items() if agent.actor is not None]

    if not affected_track_ids:
        return AttackRecord(
            attack_type="universal_perturbation",
            affected_track_ids=[],
            start_frame=start_frame_idx,
            end_frame=start_frame_idx,
            description="Attack injection failed: no agents present to perturb",
            physical_inconsistency="none (no agents present)",
            metadata={"failed": True},
        )

    perturbation = generate_perturbation(epsilon, scale_position, scale_velocity, seed=seed)
    end_frame = min(start_frame_idx + duration_frames, len(replay_state.frame_files) - 1)

    description = (
        f"A single perturbation vector (dx={perturbation['position'][0]:.3f}, "
        f"dy={perturbation['position'][1]:.3f}, dheading={perturbation['heading']:.3f} rad) was "
        f"applied uniformly to ALL {len(affected_track_ids)} tracked agents' positions and headings "
        f"starting at frame {start_frame_idx}, for {end_frame - start_frame_idx} frames. This is a "
        f"fleet-wide degradation, not targeted at any single vehicle."
    )

    physical_inconsistency = (
        "Every agent's trajectory is displaced by the identical perturbation vector at each frame -- "
        "a coordinated, uniform shift across unrelated vehicles is not physically plausible for "
        "independently-driven vehicles and differs from natural trajectory noise, which would be "
        "independent per vehicle."
    )

    return AttackRecord(
        attack_type="universal_perturbation",
        affected_track_ids=affected_track_ids,
        start_frame=start_frame_idx,
        end_frame=end_frame,
        description=description,
        physical_inconsistency=physical_inconsistency,
        metadata={
            "perturbation": perturbation,
            "epsilon": epsilon,
            "scale_position": scale_position,
            "scale_velocity": scale_velocity,
        },
    )