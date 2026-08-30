import math


def get_agent_speed(agent) -> float:
    velocity = agent.actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2)


def get_lane_id(replay_state, agent) -> int:
    world_map = replay_state.world.get_map()
    location = agent.actor.get_location()
    waypoint = world_map.get_waypoint(location)
    return waypoint.lane_id if waypoint is not None else None


def select_unique_lane_targets(replay_state, count: int, exclude_track_ids: set = None, sort_by_speed_desc: bool = False):
    exclude_track_ids = exclude_track_ids or set()

    candidates = []
    for track_id, agent in replay_state.agents.items():
        if track_id in exclude_track_ids or agent.actor is None:
            continue
        lane_id = get_lane_id(replay_state, agent)
        speed = get_agent_speed(agent)
        candidates.append((track_id, agent, lane_id, speed))

    if sort_by_speed_desc:
        candidates.sort(key=lambda c: c[3], reverse=True)

    selected = []
    used_lanes = set()
    for track_id, agent, lane_id, speed in candidates:
        if lane_id not in used_lanes:
            selected.append((track_id, agent, lane_id, speed))
            used_lanes.add(lane_id)
            if len(selected) == count:
                break

    if len(selected) < count:
        for track_id, agent, lane_id, speed in candidates:
            if len(selected) == count:
                break
            if (track_id, agent, lane_id, speed) not in selected:
                selected.append((track_id, agent, lane_id, speed))

    return selected


def release_to_autopilot(replay_state, track_id: int, traffic_manager, ignore_lights_percentage: float = 0.0):
    agent = replay_state.agents.get(track_id)
    if agent is None or agent.actor is None:
        return False

    agent.actor.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(agent.actor, ignore_lights_percentage)
    return True


def return_to_replay(replay_state, track_id: int):
    agent = replay_state.agents.get(track_id)
    if agent is None or agent.actor is None:
        return False

    agent.actor.set_autopilot(False)
    return True


def spawn_static_obstacle(replay_state, location, blueprint_filter: str = "vehicle.*"):
    import carla

    world = replay_state.world
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter(blueprint_filter)[0]

    transform = carla.Transform(location, carla.Rotation())
    actor = world.try_spawn_actor(bp, transform)

    if actor is not None:
        actor.set_simulate_physics(False)

    return actor


def offset_location_along_heading(location, yaw_radians: float, distance: float):
    import carla

    dx = math.cos(yaw_radians) * distance
    dy = math.sin(yaw_radians) * distance
    return carla.Location(x=location.x + dx, y=location.y + dy, z=location.z)