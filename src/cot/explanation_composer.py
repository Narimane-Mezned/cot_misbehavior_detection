import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cot.caption_generation import generate_cot_caption, CoTCaption


class ExplanationComposer:
    def __init__(self, pampos_target, seq_len: int = 10, dreaming_horizon: Optional[int] = None):
        self.target = pampos_target
        self.seq_len = seq_len
        self.dreaming_horizon = dreaming_horizon

    def explain(
        self,
        window,
        meta: dict,
        end_frame_objects: list,
        ego_at_end: dict,
        ego_future_obj: Optional[dict] = None,
        attack_record: Optional[object] = None,
        dreaming_errors: Optional[list] = None,
    ) -> CoTCaption:
        anomaly_score = self.target.raw_score(window)

        return generate_cot_caption(
            meta=meta,
            objects=end_frame_objects,
            ego_obj=ego_at_end,
            anomaly_score=anomaly_score,
            threshold=self.target.threshold,
            attack_record=attack_record,
            dreaming_errors=dreaming_errors,
            ego_future_obj=ego_future_obj,
        )

    def explain_if_flagged(self, window, meta: dict, end_frame_objects: list, ego_at_end: dict, **kwargs) -> Optional[CoTCaption]:
        anomaly_score = self.target.raw_score(window)
        if anomaly_score <= self.target.threshold:
            return None
        return self.explain(window, meta, end_frame_objects, ego_at_end, **kwargs)


def create_explanation_composer(pampos_target, seq_len: int = 10) -> ExplanationComposer:
    if pampos_target.threshold is None:
        raise ValueError("PAMPOSTarget must be calibrated (call .calibrate()) before creating an ExplanationComposer.")
    return ExplanationComposer(pampos_target, seq_len=seq_len)


if __name__ == "__main__":
    import numpy as np

    class FakeTarget:
        def __init__(self):
            self.threshold = 5.0

        def raw_score(self, window):
            return float(np.abs(window).sum() / window.size * 3)

    meta = {"weather": "ClearNoon", "road_type": "highway", "ego_vehicle_direction": "straight"}
    ego_obj = {"track_id": -100, "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "category": "car"}
    ego_future_calm = {"track_id": -100, "x": 10.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "category": "car"}

    objects_calm = [ego_obj, {"track_id": 1, "x": 5.0, "y": 1.0, "vx": 10.0, "vy": 0.0, "category": "car",
                               "point_count": 100, "is_camera_visible": True}]
    calm_window = np.random.normal(0, 0.5, size=(10, 8)).astype(np.float32)

    objects_severe = [ego_obj, {"track_id": 2, "x": 3.0, "y": 0.0, "vx": 20.0, "vy": 0.0, "category": "car",
                                 "point_count": 0, "is_camera_visible": False}]
    severe_window = np.random.normal(0, 5.0, size=(10, 8)).astype(np.float32)

    target = FakeTarget()
    composer = create_explanation_composer(target)

    print("=== Calm window (should NOT trigger an explanation) ===")
    result = composer.explain_if_flagged(calm_window, meta, objects_calm, ego_obj, ego_future_obj=ego_future_calm)
    print(result if result else "No explanation composed (window not flagged) -- correct, this is the expected behavior for a live composer.")

    print("\n=== Severe window (SHOULD trigger an explanation) ===")
    result = composer.explain_if_flagged(severe_window, meta, objects_severe, ego_obj, ego_future_obj=ego_future_calm)
    if result:
        print(result.full_caption)
    else:
        print("No explanation composed -- unexpected for this test case.")