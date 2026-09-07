import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.deepaccident_loader import (
    parse_label_file,
    parse_meta,
    list_scenarios,
    get_frame_number,
    EGO_TRACK_ID,
)


@dataclass
class ReplayAgent:
    track_id: int
    category: str
    actor = None
    is_ego: bool = False


@dataclass
class ReplayState:
    world: object
    agents: dict = field(default_factory=dict)
    town: str = ""
    frame_files: list = field(default_factory=list)
    current_frame_idx: int = 0


def get_town_from_scenario_name(scenario_name: str) -> str:
    return scenario_name.split("_")[0]


def connect_carla(host: str = "localhost", port: int = 2000, timeout: float = 10.0):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def load_scenario_world(client, scenario_type_dir: Path, scenario_name: str, low_resource_mode: bool = True):
    import carla
    town = get_town_from_scenario_name(scenario_name)
    world = client.load_world(town)

    if low_resource_mode:
        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)

    label_dir = Path(scenario_type_dir) / "ego_vehicle" / "label" / scenario_name
    frame_files = sorted(label_dir.glob("*.txt"), key=lambda p: get_frame_number(p.name))

    meta_path = Path(scenario_type_dir) / "meta" / f"{scenario_name}.txt"
    meta = parse_meta(meta_path)

    replay_state = ReplayState(world=world, town=town, frame_files=frame_files)
    return replay_state, meta


def spawn_agents_at_frame(replay_state: ReplayState, frame_idx: int):
    import carla
    world = replay_state.world
    blueprint_library = world.get_blueprint_library()
    parsed = parse_label_file(replay_state.frame_files[frame_idx])

    for obj in parsed["objects"]:
        if obj["track_id"] in replay_state.agents:
            continue

        if obj["category"] == "car":
            bp = blueprint_library.filter("vehicle.*")[0]
        elif obj["category"] == "truck":
            bp = blueprint_library.filter("vehicle.*truck*")
            bp = bp[0] if len(bp) > 0 else blueprint_library.filter("vehicle.*")[0]
        elif obj["category"] == "pedestrian":
            bp = blueprint_library.filter("walker.pedestrian.*")[0]
        else:
            bp = blueprint_library.filter("vehicle.*")[0]

        transform = carla.Transform(
            carla.Location(x=obj["x"], y=obj["y"], z=obj["z"] + 0.5),
            carla.Rotation(yaw=obj["yaw"] * 180.0 / 3.14159265),
        )

        actor = world.try_spawn_actor(bp, transform)
        if actor is None:
            continue

        agent = ReplayAgent(
            track_id=obj["track_id"],
            category=obj["category"],
            is_ego=(obj["track_id"] == EGO_TRACK_ID),
        )
        agent.actor = actor
        replay_state.agents[obj["track_id"]] = agent

    replay_state.current_frame_idx = frame_idx


def apply_frame_state(replay_state: ReplayState, frame_idx: int, skip_track_ids: set = None):
    import carla
    skip_track_ids = skip_track_ids or set()
    parsed = parse_label_file(replay_state.frame_files[frame_idx])

    for obj in parsed["objects"]:
        if obj["track_id"] in skip_track_ids:
            continue
        agent = replay_state.agents.get(obj["track_id"])
        if agent is None or agent.actor is None:
            continue

        transform = carla.Transform(
            carla.Location(x=obj["x"], y=obj["y"], z=obj["z"] + 0.5),
            carla.Rotation(yaw=obj["yaw"] * 180.0 / 3.14159265),
        )
        agent.actor.set_transform(transform)

    replay_state.current_frame_idx = frame_idx


def cleanup_replay(replay_state: ReplayState):
    for agent in replay_state.agents.values():
        if agent.actor is not None:
            try:
                if agent.actor.is_alive:
                    agent.actor.destroy()
            except RuntimeError:
                pass
    replay_state.agents.clear()


def cleanup_actor_ids(world, actor_ids: list):
    for actor_id in actor_ids:
        actor = world.get_actor(actor_id)
        if actor is not None:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass


def cleanup_attack_record(replay_state: ReplayState, attack_record):
    metadata = getattr(attack_record, "metadata", {}) or {}
    for key in ("obstacle_actor_ids", "emergency_actor_ids", "sybil_actor_ids"):
        if key in metadata:
            cleanup_actor_ids(replay_state.world, metadata[key])