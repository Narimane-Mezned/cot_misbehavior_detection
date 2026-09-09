import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

FEATURE_NAMES = ["x", "y", "vx", "vy", "yaw", "point_count", "is_camera_visible", "distance_to_ego"]

FEATURE_PLAIN = {
    "x": "lateral position",
    "y": "longitudinal position",
    "vx": "lateral velocity",
    "vy": "longitudinal velocity",
    "yaw": "heading",
    "point_count": "LiDAR return count",
    "is_camera_visible": "camera visibility",
    "distance_to_ego": "distance to ego",
}

MOTION_FEATURES = {"x", "y", "vx", "vy", "yaw"}
BINARY_FEATURES = {"is_camera_visible"}
SENSOR_FEATURES = {"point_count", "is_camera_visible"}


@dataclass
class CoTCaption:
    subject: str
    verdict: str
    evidence: str
    sensor_corroboration: str
    context: str
    attack_note: str
    risk_level: str
    detector_status: str
    full_caption: str
    metadata: Dict = field(default_factory=dict)


def distance_between(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def _variation_index(seed_parts: tuple, n: int) -> int:
    return hash(seed_parts) % n


def find_subject(objects: List[dict], subject_track_id: int) -> Optional[dict]:
    for obj in objects:
        if obj["track_id"] == subject_track_id:
            return obj
    return None


def describe_subject(subject: Optional[dict], subject_track_id: int, ego_obj: dict) -> str:
    if subject is None:
        return (f"Subject: track_id={subject_track_id}, not present in the final frame of "
                f"this window (left sensor range or the scene).")

    distance = distance_between(subject, ego_obj)
    return (f"Subject: {subject['category']} (track_id={subject_track_id}), "
            f"{distance:.1f}m from ego.")


def state_verdict(anomaly_score: float, threshold: float, seed: tuple) -> tuple:
    ratio = anomaly_score / threshold if threshold > 0 else 0.0

    if anomaly_score <= threshold:
        level = "low"
        options = [
            f"Not flagged: predicted-versus-observed motion error ({anomaly_score:.3f}) "
            f"stays below the calibrated threshold ({threshold:.3f}).",
            f"No misbehaviour indicated -- deviation score {anomaly_score:.3f} against a "
            f"threshold of {threshold:.3f}.",
        ]
    elif ratio < 2.0:
        level = "moderate"
        options = [
            f"Flagged as borderline: deviation score {anomaly_score:.3f} exceeds the "
            f"calibrated threshold ({threshold:.3f}) by {ratio:.1f}x.",
            f"Mildly suspicious -- {anomaly_score:.3f} against a threshold of "
            f"{threshold:.3f} ({ratio:.1f}x).",
        ]
    else:
        level = "high"
        options = [
            f"Flagged as strongly suspicious: deviation score {anomaly_score:.3f} is "
            f"{ratio:.1f}x the calibrated threshold ({threshold:.3f}).",
            f"Strong indication of misreported state -- {anomaly_score:.3f}, "
            f"{ratio:.1f}x the threshold of {threshold:.3f}.",
        ]

    return level, options[_variation_index(seed, len(options))]


CLOSE_TRACKING_MAX_RESIDUAL = 1.5


def describe_evidence(feature_errors: Optional[List[float]], is_flagged: bool = True, top_k: int = 3) -> tuple:
    if feature_errors is None or len(feature_errors) != len(FEATURE_NAMES):
        return ("", [])

    pairs = list(zip(FEATURE_NAMES, list(feature_errors)))
    pairs.sort(key=lambda p: p[1], reverse=True)
    top = pairs[:top_k]
    largest_name, largest_err = top[0]

    if not is_flagged:
        parts = [f"{FEATURE_PLAIN[name]} ({err:.2f}x)" for name, err in top]
        if largest_err <= CLOSE_TRACKING_MAX_RESIDUAL:
            if largest_name in BINARY_FEATURES:
                detail = f"{FEATURE_PLAIN[largest_name]}, which is a binary flag"
            else:
                detail = f"{FEATURE_PLAIN[largest_name]} at {largest_err:.2f}x its typical error"
            return (f"All features track their predicted values closely; the largest residual is "
                    f"{detail}.", [name for name, _ in top])
        return (f"Largest residuals are {', '.join(parts)} -- above average but within the range "
                f"seen in normal driving, which is why the aggregate stays below threshold.",
                [name for name, _ in top])

    parts = []
    for name, err in top:
        if name in BINARY_FEATURES:
            parts.append(f"{FEATURE_PLAIN[name]} (flipped against prediction)")
        else:
            parts.append(f"{FEATURE_PLAIN[name]} ({err:.2f}x)")

    dominant = top[0][0]
    if dominant in MOTION_FEATURES:
        lead = ("The deviation comes from motion the model could not predict from this "
                "vehicle's own preceding trajectory")
    elif dominant in SENSOR_FEATURES:
        lead = ("The deviation is in how this vehicle's sensor signature changed across the "
                "window rather than in its motion -- the readings varied far more than "
                "predicted")
    else:
        lead = "The deviation comes from"

    return (f"{lead}: {', '.join(parts)} -- these are per-feature ratios, not the "
            f"aggregate's ratio to threshold.", [name for name, _ in top])


SPARSE_LIDAR_THRESHOLD = 50
LIDAR_MAX_RANGE_M = 80.0
SPARSE_EXPECTED_BEYOND_M = 30.0
LIDAR_CONFIDENT_RANGE_M = 50.0


def describe_sensor_corroboration(subject: Optional[dict], ego_obj: dict,
                                  top_features: Optional[List[str]] = None,
                                  is_flagged: bool = True) -> str:
    if subject is None:
        return ""

    sensor_is_the_anomaly = bool(top_features) and top_features[0] in SENSOR_FEATURES

    distance = distance_between(subject, ego_obj)
    point_count = subject["point_count"]
    camera_visible = subject["is_camera_visible"]

    if point_count == 0:
        if distance > LIDAR_MAX_RANGE_M:
            return (f"Sensor corroboration is unavailable at this range: no LiDAR returns at "
                    f"{distance:.1f}m, which is expected beyond the sensor's effective range "
                    f"(~{LIDAR_MAX_RANGE_M:.0f}m). Absence of returns here is not itself "
                    f"suspicious.")
        if distance > LIDAR_CONFIDENT_RANGE_M:
            hedge = (f"toward the edge of reliable LiDAR range, so this may reflect range "
                     f"limits rather than a genuine inconsistency")
        else:
            hedge = (f"well within reliable range, so this object's claimed presence is "
                     f"unsupported by direct observation")
        tail = ("" if is_flagged else
                " The aggregate score does not flag this window, so this is noted as an "
                "observation rather than a finding.")
        return (f"Sensor corroboration is absent: LiDAR returned no points at {distance:.1f}m, "
                f"{hedge}.{tail}")

    if point_count < SPARSE_LIDAR_THRESHOLD:
        visibility = "and no camera sees it" if not camera_visible else "though a camera does see it"
        if distance > SPARSE_EXPECTED_BEYOND_M:
            return (f"Sensor corroboration is thin but unremarkable: {point_count} LiDAR points at "
                    f"{distance:.1f}m {visibility}. Return density falls off with range, so a low "
                    f"count at this distance is expected rather than suspicious.")
        tail = ("" if is_flagged else
                " The aggregate score does not flag this window, so this is noted as an "
                "observation rather than a finding.")
        return (f"Sensor corroboration is sparse: only {point_count} LiDAR points at "
                f"{distance:.1f}m {visibility}, within the range where a denser return would "
                f"be expected.{tail}")

    if not camera_visible and distance < 30.0:
        return (f"Sensor corroboration is inconsistent: LiDAR returns are strong "
                f"({point_count} points) yet no camera sees this object at only "
                f"{distance:.1f}m -- the two sensing modalities disagree.")

    if not camera_visible:
        return (f"Sensor corroboration is limited: {point_count} LiDAR points, no camera "
                f"visibility at {distance:.1f}m (plausible at this range).")

    if sensor_is_the_anomaly:
        return (f"Present sensing quality is good: {point_count} LiDAR points and confirmed "
                f"camera visibility at {distance:.1f}m. The anomaly is not weak sensing but the "
                f"unexplained variation in these readings across the window.")

    return (f"Sensor corroboration is consistent: {point_count} LiDAR points and confirmed "
            f"camera visibility at {distance:.1f}m, so the deviation cannot be attributed "
            f"to poor observation.")


def describe_context(meta: dict, num_agents: int) -> str:
    bits = []
    if meta.get("road_type"):
        bits.append(meta["road_type"])
    if meta.get("weather"):
        bits.append(meta["weather"])
    if not bits:
        return ""
    plural = "agent" if num_agents == 1 else "agents"
    return f"Context: {', '.join(bits)}, {num_agents} other tracked {plural}."


def describe_attack(attack_record: Optional[object],
                    clean_score: Optional[float],
                    poisoned_score: Optional[float],
                    threshold: float) -> tuple:
    if attack_record is None:
        return "", None

    attack_type = getattr(attack_record, "attack_type", "unknown")

    if clean_score is not None and poisoned_score is not None:
        was_flagged = clean_score > threshold
        now_flagged = poisoned_score > threshold

        if was_flagged and not now_flagged:
            severity = clean_score / threshold if threshold > 0 else 0.0
            if severity >= 2.0:
                headline = ("WARNING -- DETECTION COMPROMISED. A strongly suspicious vehicle has "
                            "been hidden from the detector.")
            else:
                headline = ("CAUTION -- DETECTION COMPROMISED. A borderline detection has been "
                            "suppressed; the vehicle itself was only mildly suspicious, but the "
                            "calibration is nonetheless tampered with.")
            return (f"{headline} A {attack_type} poisoned the calibration data: the deployed "
                    f"detector reports {poisoned_score:.3f} (apparently safe) but the untampered "
                    f"baseline gives {clean_score:.3f}. Recalibrate from clean data before "
                    f"relying on this detector.",
                    "compromised")

        if poisoned_score > clean_score:
            return (f"A {attack_type} was applied to the calibration data and did not succeed: "
                    f"the score moved from {clean_score:.3f} to {poisoned_score:.3f}, making "
                    f"detection more likely rather than less. This attack applies a uniform "
                    f"shift across all calibration data and does not target any individual "
                    f"vehicle.", None)

        return (f"A {attack_type} was applied to the calibration data; the score moved from "
                f"{clean_score:.3f} to {poisoned_score:.3f} and the detection outcome is "
                f"unchanged.", None)

    inconsistency = getattr(attack_record, "physical_inconsistency", None)
    if inconsistency:
        return f"Injected attack ({attack_type}): {inconsistency}", None
    return f"An attack of type {attack_type} was injected into this scene.", None


def describe_ego_response(ego_current: dict, ego_future: Optional[dict]) -> str:
    if ego_future is None:
        return ""

    current_speed = math.sqrt(ego_current["vx"] ** 2 + ego_current["vy"] ** 2)
    future_speed = math.sqrt(ego_future["vx"] ** 2 + ego_future["vy"] ** 2)
    lateral_shift = abs(ego_future["y"] - ego_current["y"])

    if current_speed < 0.5 and future_speed < 0.5:
        return "Ego is stationary."
    if future_speed - current_speed < -1.0:
        return f"Ego response: braking, {current_speed:.1f} to {future_speed:.1f} m/s."
    if future_speed - current_speed > 1.0:
        return f"Ego response: accelerating, {current_speed:.1f} to {future_speed:.1f} m/s."
    if lateral_shift > 1.5:
        return f"Ego response: lateral shift of {lateral_shift:.1f}m (yield or swerve)."
    if current_speed < 0.5:
        return f"Ego response: none, barely moving ({current_speed:.1f} m/s)."
    return f"Ego response: none, holding {current_speed:.1f} m/s."


def generate_cot_caption(
    meta: dict,
    objects: List[dict],
    ego_obj: dict,
    anomaly_score: float,
    threshold: float,
    subject_track_id: Optional[int] = None,
    feature_errors: Optional[List[float]] = None,
    attack_record: Optional[object] = None,
    clean_score: Optional[float] = None,
    poisoned_score: Optional[float] = None,
    ego_future_obj: Optional[dict] = None,
    dreaming_errors: Optional[List[float]] = None,
) -> CoTCaption:

    subject = find_subject(objects, subject_track_id) if subject_track_id is not None else None
    subject_text = describe_subject(subject, subject_track_id, ego_obj) if subject_track_id is not None else ""

    seed = (round(anomaly_score, 4), subject_track_id or 0)

    true_score = clean_score if clean_score is not None else anomaly_score
    risk_level, verdict_text = state_verdict(true_score, threshold, seed)

    if clean_score is not None and poisoned_score is not None and clean_score != poisoned_score:
        verdict_text = "True assessment (untampered baseline): " + verdict_text

    evidence_text, top_features = describe_evidence(feature_errors, is_flagged=(true_score > threshold))

    if evidence_text and clean_score is not None and poisoned_score is not None:
        evidence_text = evidence_text
    sensor_text = describe_sensor_corroboration(subject, ego_obj, top_features,
                                                is_flagged=(true_score > threshold))
    context_text = describe_context(meta, max(0, len(objects) - 1))
    attack_text, override_level = describe_attack(attack_record, clean_score, poisoned_score, threshold)
    ego_text = describe_ego_response(ego_obj, ego_future_obj)

    detector_status = override_level if override_level == "compromised" else "intact"

    dreaming_text = ""
    if dreaming_errors and len(dreaming_errors) >= 2 and dreaming_errors[0] > 0:
        growth = dreaming_errors[-1] / dreaming_errors[0]
        if growth > 3.0:
            dreaming_text = (f"Projected forward, the error compounds from "
                             f"{dreaming_errors[0]:.3f} to {dreaming_errors[-1]:.3f} over "
                             f"{len(dreaming_errors)} steps.")

    ordered = []
    if attack_text and detector_status == "compromised":
        ordered.append(attack_text)
    if subject_text:
        ordered.append(subject_text)
    ordered.append(verdict_text)
    if evidence_text:
        ordered.append(evidence_text)
    if sensor_text:
        ordered.append(sensor_text)
    if dreaming_text:
        ordered.append(dreaming_text)
    if attack_text and detector_status != "compromised":
        ordered.append(attack_text)
    if context_text:
        ordered.append(context_text)
    if ego_text:
        ordered.append(ego_text)

    return CoTCaption(
        subject=subject_text,
        verdict=verdict_text,
        evidence=evidence_text,
        sensor_corroboration=sensor_text,
        context=context_text,
        attack_note=attack_text,
        risk_level=risk_level,
        detector_status=detector_status,
        full_caption=" ".join(ordered),
        metadata={
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "subject_track_id": subject_track_id,
            "top_features": top_features,
            "attack_type": getattr(attack_record, "attack_type", None) if attack_record else None,
            "clean_score": clean_score,
            "poisoned_score": poisoned_score,
        },
    )


if __name__ == "__main__":
    meta = {"weather": "MidRainSunset", "road_type": "four-way junction"}
    ego = {"track_id": -100, "x": 0.0, "y": 0.0, "vx": 5.3, "vy": 0.0, "category": "car"}
    subject = {"track_id": 7109, "x": 7.0, "y": 0.0, "vx": 2.0, "vy": 1.0,
               "category": "truck", "point_count": 1500, "is_camera_visible": True}
    other = {"track_id": 7108, "x": 20.0, "y": 3.0, "vx": 4.0, "vy": 0.0,
             "category": "car", "point_count": 900, "is_camera_visible": True}
    objects = [ego, subject, other]

    print("=" * 78)
    print("CASE 1 -- P2 fix: high score, motion-driven (previously explained nothing)")
    print("=" * 78)
    print(generate_cot_caption(meta, objects, ego, 87.648, 8.240, subject_track_id=7109,
                               feature_errors=[0.9, 1.1, 4.2, 3.1, 3.8, 0.3, 0.2, 0.7],
                               ego_future_obj=ego).full_caption)

    print("\n" + "=" * 78)
    print("CASE 2 -- P4 fix: camera-invisible but 13278 LiDAR points")
    print("=" * 78)
    subject2 = dict(subject, point_count=13278, is_camera_visible=False, x=3.5)
    print(generate_cot_caption(meta, [ego, subject2, other], ego, 53.906, 8.240,
                               subject_track_id=7109,
                               feature_errors=[0.5, 0.6, 2.9, 2.2, 1.8, 0.4, 3.5, 0.6],
                               ego_future_obj=ego).full_caption)

    print("\n" + "=" * 78)
    print("CASE 3 -- P8 fix: backdoor succeeded (previously 'low risk')")
    print("=" * 78)
    class Rec:
        attack_type = "backdoor_attack"
        physical_inconsistency = "..."
    c3 = generate_cot_caption(meta, objects, ego, 1.068, 8.240, subject_track_id=7109,
                              feature_errors=[0.1, 0.1, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
                              attack_record=Rec(), clean_score=87.648, poisoned_score=1.068,
                              ego_future_obj=ego)
    print(f"risk_level = {c3.risk_level}")
    print(c3.full_caption)

    print("\n" + "=" * 78)
    print("CASE 4 -- P10 fix: clean-label failed (previously 'flagged by attack')")
    print("=" * 78)
    class Rec2:
        attack_type = "clean_label_feature_collision"
        physical_inconsistency = "..."
    c4 = generate_cot_caption(meta, objects, ego, 93.228, 8.240, subject_track_id=7109,
                              feature_errors=[1.0, 1.2, 4.5, 3.3, 4.0, 0.3, 0.2, 0.8],
                              attack_record=Rec2(), clean_score=87.648, poisoned_score=93.228,
                              ego_future_obj=ego)
    print(f"risk_level = {c4.risk_level}")
    print(c4.full_caption)

    print("\n" + "=" * 78)
    print("CASE 5 -- P9 fix: stationary ego")
    print("=" * 78)
    ego_still = dict(ego, vx=0.0, vy=0.0)
    print(generate_cot_caption(meta, [ego_still, subject, other], ego_still, 0.547, 8.240,
                               subject_track_id=7109,
                               feature_errors=[0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
                               ego_future_obj=ego_still).full_caption)