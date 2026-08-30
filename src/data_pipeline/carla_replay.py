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


def load_scenario_world(client, scenario_type_dir: Path, scenario_name: str):
    import carla

    town = get_town_from_scenario_name(scenario_name)
    world = client.load_world(town)

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
            agent.actor.destroy()
    replay_state.agents.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--scenario_type", type=str, default="type1_subtype1_normal")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    scenario_type_dir = Path(args.data_root) / args.scenario_type
    scenario_names = list_scenarios(scenario_type_dir)
    scenario_name = scenario_names[0]
    print(f"[replay] Loading scenario: {scenario_name}")

    client = connect_carla(host=args.host, port=args.port)
    replay_state, meta = load_scenario_world(client, scenario_type_dir, scenario_name)
    print(f"[replay] Loaded town: {replay_state.town}")
    print(f"[replay] Meta: {meta}")
    print(f"[replay] Total frames: {len(replay_state.frame_files)}")

    spawn_agents_at_frame(replay_state, frame_idx=0)
    print(f"[replay] Spawned {len(replay_state.agents)} agents at frame 0")

    for frame_idx in range(1, min(len(replay_state.frame_files), 20)):
        apply_frame_state(replay_state, frame_idx)

    print(f"[replay] Replayed to frame {replay_state.current_frame_idx}")

    cleanup_replay(replay_state)
    print("[replay] Cleaned up actors")