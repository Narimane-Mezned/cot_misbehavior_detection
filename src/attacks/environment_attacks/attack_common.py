import math


def get_agent_speed(agent, replay_state=None, frame_idx=None) -> float:
    """Replayed agents are positioned with set_transform rather than driven, so
    the physics velocity CARLA reports for them is identically zero. Speed is
    therefore read from the ground-truth label file for the current frame, and
    falls back to the physics velocity only when that is unavailable."""
    speed = recorded_speed(replay_state, getattr(agent, "track_id", None), frame_idx)
    if speed == speed:                      # not NaN
        return speed
    velocity = agent.actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2)


def recorded_speed(replay_state, track_id, frame_idx=None) -> float:
    """Ground-truth speed of one track at a frame, taken from the label file's
    vx and vy fields. Returns NaN when it cannot be determined."""
    if replay_state is None or track_id is None:
        return float("nan")
    files = getattr(replay_state, "frame_files", None)
    if not files:
        return float("nan")
    if frame_idx is None:
        frame_idx = getattr(replay_state, "current_frame_idx", 0)
    if not (0 <= frame_idx < len(files)):
        return float("nan")
    try:
        from src.data_pipeline.deepaccident_loader import parse_label_file
        parsed = parse_label_file(files[frame_idx])
    except Exception:
        return float("nan")
    for obj in parsed.get("objects", []):
        if str(obj.get("track_id")) == str(track_id):
            return math.hypot(float(obj.get("vx", 0.0)), float(obj.get("vy", 0.0)))
    return float("nan")


def get_lane_id(replay_state, agent) -> int:
    world_map = replay_state.world.get_map()
    location = agent.actor.get_location()
    waypoint = world_map.get_waypoint(location)
    return waypoint.lane_id if waypoint is not None else None


UNTRACKED_TRACK_ID = -1
EGO_TRACK_ID = -100
MIN_TARGET_SPEED_MS = 1.0


def select_unique_lane_targets(replay_state, count: int, exclude_track_ids: set = None,
                               sort_by_speed_desc: bool = False,
                               min_speed_ms: float = MIN_TARGET_SPEED_MS,
                               frame_idx: int = None):
    exclude_track_ids = set(exclude_track_ids or set())
    exclude_track_ids.update({UNTRACKED_TRACK_ID, EGO_TRACK_ID,
                              str(UNTRACKED_TRACK_ID), str(EGO_TRACK_ID)})

    candidates = []
    skipped_stationary = 0
    for track_id, agent in replay_state.agents.items():
        if track_id in exclude_track_ids or agent.actor is None:
            continue
        speed = get_agent_speed(agent, replay_state, frame_idx)
        if speed < min_speed_ms:
            skipped_stationary += 1
            continue
        lane_id = get_lane_id(replay_state, agent)
        candidates.append((track_id, agent, lane_id, speed))

    if skipped_stationary:
        print(f"[targets] skipped {skipped_stationary} agent(s) below "
              f"{min_speed_ms} m/s -- an attack cannot slow a stationary vehicle")
    if not candidates and skipped_stationary:
        print(f"[targets] WARNING: every agent was judged stationary. During "
              f"replay the physics velocity is zero, so this means the recorded "
              f"trajectory was unavailable and the fallback was used.")

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

    for z_offset in (0.0, 0.5, 1.0):
        transform = carla.Transform(
            carla.Location(x=location.x, y=location.y, z=location.z + z_offset),
            carla.Rotation(),
        )
        actor = world.try_spawn_actor(bp, transform)
        if actor is not None:
            actor.set_simulate_physics(False)
            return actor

    return None


def offset_location_along_heading(location, yaw_radians: float, distance: float):
    import carla

    dx = math.cos(yaw_radians) * distance
    dy = math.sin(yaw_radians) * distance
    return carla.Location(x=location.x + dx, y=location.y + dy, z=location.z)

def enforce_speed(replay_state, track_id: int, target_speed_ms: float):
    import carla

    agent = replay_state.agents.get(track_id)
    if agent is None or agent.actor is None:
        return False

    try:
        if not agent.actor.is_alive:
            return False
        if not agent.actor.type_id.startswith("vehicle."):
            return False
        transform = agent.actor.get_transform()
        forward = transform.get_forward_vector()
        agent.actor.set_target_velocity(
            carla.Vector3D(
                x=forward.x * target_speed_ms,
                y=forward.y * target_speed_ms,
                z=0.0,
            )
        )
        if target_speed_ms <= 0.01:
            agent.actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        return True
    except RuntimeError:
        return False


def enforced_behaviour(track_ids, target_speed_ms: float, description: str):
    return {
        "track_ids": list(track_ids),
        "target_speed_ms": float(target_speed_ms),
        "description": description,
    }