import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CoTCaption:
    scene_description: str
    critical_objects: List[Dict]
    risk_level: str
    risk_explanation: str
    counterfactual: str
    action_plan: str
    full_caption: str
    metadata: Dict = field(default_factory=dict)


def distance_between(obj_a: dict, obj_b: dict) -> float:
    return math.sqrt((obj_a["x"] - obj_b["x"]) ** 2 + (obj_a["y"] - obj_b["y"]) ** 2)


def describe_scene(meta: dict, num_agents: int) -> str:
    road_type = meta.get("road_type", "an unspecified road type")
    weather = meta.get("weather", "unspecified weather")
    ego_direction = meta.get("ego_vehicle_direction", "an unspecified direction")

    return (
        f"The ego vehicle is navigating {road_type} under {weather} conditions, "
        f"heading {ego_direction}, with {num_agents} other tracked agent(s) nearby."
    )


def identify_critical_objects(
    objects: List[dict],
    ego_obj: dict,
    attack_record: Optional[object] = None,
    top_k: int = 3,
) -> List[Dict]:
    affected_ids = set(getattr(attack_record, "affected_track_ids", []) or [])

    candidates = []
    for obj in objects:
        if obj["track_id"] == ego_obj["track_id"]:
            continue

        distance = distance_between(obj, ego_obj)
        is_flagged_by_attack = obj["track_id"] in affected_ids
        sensor_inconsistent = (obj["point_count"] == 0) or (not obj["is_camera_visible"] and distance < 30.0)

        reason_parts = []
        if is_flagged_by_attack:
            reason_parts.append(f"flagged by injected attack ({getattr(attack_record, 'attack_type', 'unknown')})")
        if sensor_inconsistent:
            reason_parts.append(
                f"sensor evidence is weak (point_count={obj['point_count']}, "
                f"is_camera_visible={obj['is_camera_visible']}) despite {distance:.1f}m proximity"
            )
        if not reason_parts:
            reason_parts.append(f"{distance:.1f}m from ego, no sensor inconsistency observed")

        candidates.append({
            "track_id": obj["track_id"],
            "category": obj["category"],
            "distance_to_ego": distance,
            "is_flagged_by_attack": is_flagged_by_attack,
            "reason": "; ".join(reason_parts),
        })

    candidates.sort(key=lambda c: (not c["is_flagged_by_attack"], c["distance_to_ego"]))
    return candidates[:top_k]


def predict_risk(
    anomaly_score: float,
    threshold: float,
    dreaming_errors: Optional[List[float]] = None,
) -> Dict[str, str]:
    is_anomalous = anomaly_score > threshold
    ratio = anomaly_score / threshold if threshold > 0 else 0.0

    if not is_anomalous:
        level = "low"
        explanation = (
            f"Anomaly score ({anomaly_score:.3f}) is below the calibrated threshold "
            f"({threshold:.3f}) -- behavior is consistent with normal driving patterns."
        )
    elif ratio < 2.0:
        level = "moderate"
        explanation = (
            f"Anomaly score ({anomaly_score:.3f}) exceeds the calibrated threshold "
            f"({threshold:.3f}) by a moderate margin, suggesting a deviation worth monitoring."
        )
    else:
        level = "high"
        explanation = (
            f"Anomaly score ({anomaly_score:.3f}) substantially exceeds the calibrated threshold "
            f"({threshold:.3f}) ({ratio:.1f}x), indicating a strong deviation from normal patterns."
        )

    if dreaming_errors and len(dreaming_errors) >= 2:
        growth = dreaming_errors[-1] / dreaming_errors[0] if dreaming_errors[0] > 0 else 0.0
        if growth > 3.0:
            explanation += (
                f" Multi-step projection shows error compounding rapidly ({dreaming_errors[0]:.3f} -> "
                f"{dreaming_errors[-1]:.3f} over {len(dreaming_errors)} steps), suggesting the deviation "
                f"would worsen if unaddressed."
            )

    return {"level": level, "explanation": explanation}


def counterfactual_reasoning(attack_record: Optional[object] = None) -> str:
    if attack_record is None:
        return "No injected attack is present in this scene; no counterfactual is needed."

    physical_inconsistency = getattr(attack_record, "physical_inconsistency", None)
    if physical_inconsistency:
        return physical_inconsistency

    return "An attack was injected but no specific physical inconsistency was recorded."


def plan_action(ego_current: dict, ego_future: Optional[dict] = None) -> str:
    if ego_future is None:
        return "No future ego state available to determine planned action."

    current_speed = math.sqrt(ego_current["vx"] ** 2 + ego_current["vy"] ** 2)
    future_speed = math.sqrt(ego_future["vx"] ** 2 + ego_future["vy"] ** 2)
    speed_delta = future_speed - current_speed

    lateral_shift = abs(ego_future["y"] - ego_current["y"])

    if speed_delta < -1.0:
        return f"Ego vehicle is expected to brake, reducing speed from {current_speed:.1f} to {future_speed:.1f} m/s."
    elif lateral_shift > 1.5:
        return f"Ego vehicle is expected to yield or swerve, shifting laterally by {lateral_shift:.1f}m."
    else:
        return f"Ego vehicle is expected to continue at a steady speed (~{current_speed:.1f} m/s), no evasive action planned."


def generate_cot_caption(
    meta: dict,
    objects: List[dict],
    ego_obj: dict,
    anomaly_score: float,
    threshold: float,
    attack_record: Optional[object] = None,
    dreaming_errors: Optional[List[float]] = None,
    ego_future_obj: Optional[dict] = None,
) -> CoTCaption:
    scene_desc = describe_scene(meta, num_agents=len(objects) - 1)
    critical_objects = identify_critical_objects(objects, ego_obj, attack_record)
    risk = predict_risk(anomaly_score, threshold, dreaming_errors)
    counterfactual = counterfactual_reasoning(attack_record)
    action = plan_action(ego_obj, ego_future_obj)

    critical_summary = "; ".join(
        f"{c['category']} (track_id={c['track_id']}, {c['reason']})" for c in critical_objects
    ) if critical_objects else "no other agents in range"

    full_caption = (
        f"{scene_desc} "
        f"Critical objects: {critical_summary}. "
        f"Risk assessment ({risk['level']}): {risk['explanation']} "
        f"{counterfactual} "
        f"{action}"
    )

    return CoTCaption(
        scene_description=scene_desc,
        critical_objects=critical_objects,
        risk_level=risk["level"],
        risk_explanation=risk["explanation"],
        counterfactual=counterfactual,
        action_plan=action,
        full_caption=full_caption,
        metadata={
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "attack_type": getattr(attack_record, "attack_type", None) if attack_record else None,
        },
    )


if __name__ == "__main__":
    meta = {
        "weather": "ClearNoon",
        "road_type": "three-way junction",
        "ego_vehicle_direction": "right",
        "other_vehicle_direction": "straight",
    }

    ego_obj = {"track_id": -100, "x": 0.0, "y": 0.0, "vx": 5.0, "vy": 0.0, "category": "car"}
    ego_future_braking = {"track_id": -100, "x": 2.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "category": "car"}

    objects = [
        ego_obj,
        {"track_id": 7255, "x": 15.0, "y": 2.0, "vx": 2.0, "vy": 0.0, "category": "car",
         "point_count": 45, "is_camera_visible": True},
        {"track_id": -1010, "x": -20.0, "y": 0.0, "vx": 15.0, "vy": 0.0, "category": "car",
         "point_count": 0, "is_camera_visible": False},
    ]

    class FakeAttackRecord:
        attack_type = "fake_emergency"
        affected_track_ids = [7255]
        physical_inconsistency = (
            "The emergency vehicle has no prior trajectory history before this frame -- it appeared "
            "already at full speed with lights active, inconsistent with gradual sensor-range entry."
        )

    print("=== BENIGN SCENARIO ===\n")
    benign_caption = generate_cot_caption(
        meta, objects[:2], ego_obj,
        anomaly_score=0.8, threshold=2.8,
        ego_future_obj=ego_obj,
    )
    print(benign_caption.full_caption)

    print("\n\n=== ATTACK SCENARIO (fake_emergency) ===\n")
    attack_caption = generate_cot_caption(
        meta, objects, ego_obj,
        anomaly_score=9.4, threshold=2.8,
        attack_record=FakeAttackRecord(),
        dreaming_errors=[0.05, 0.13, 0.25, 0.38, 0.52],
        ego_future_obj=ego_future_braking,
    )
    print(attack_caption.full_caption)
    print(f"\nRisk level: {attack_caption.risk_level}")
    print(f"Critical objects: {attack_caption.critical_objects}")