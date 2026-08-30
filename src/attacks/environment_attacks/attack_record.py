from dataclasses import dataclass, field


@dataclass
class AttackRecord:
    attack_type: str
    affected_track_ids: list
    start_frame: int
    end_frame: int
    description: str
    physical_inconsistency: str
    metadata: dict = field(default_factory=dict)