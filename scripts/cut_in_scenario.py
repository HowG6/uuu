import carla


# [CUT_IN_SCENARIO] All distances and speeds for the reproducible experiment
# live here so UDMC and baseline entry points can share exactly the same setup.
CUT_IN_CONFIG = {
    'ego_initial_speed': 30.0,       # km/h
    'front_gap': 25.0,              # m ahead of ego, in the right lane
    'rear_gap': 15.0,               # m behind ego, in the right lane
    'front_speed': 25.0,            # km/h
    'rear_initial_speed': 30.0,      # km/h
    'rear_overtake_speed': 50.0,     # km/h before/during lane change
    'rear_settled_speed': 38.0,      # km/h after lane change
    'trigger_from_rear_spawn': 25.0, # m; fixed map position, not ego-relative
    'lane_change_length': 25.0,      # m
}


def _single_waypoint(waypoints, description):
    if not waypoints:
        raise ValueError('Unable to find {}'.format(description))
    return waypoints[0]


def _raised_transform(waypoint, height_offset=0.2):
    transform = waypoint.transform
    return carla.Transform(
        carla.Location(x=transform.location.x,
                       y=transform.location.y,
                       z=transform.location.z + height_offset),
        carla.Rotation(pitch=transform.rotation.pitch,
                       yaw=transform.rotation.yaw,
                       roll=transform.rotation.roll))


def _validate_lane_change_corridor(trigger_wp, length):
    # [CUT_IN_SCENARIO] Fail during scenario construction instead of waiting
    # until rear_sv reaches the trigger several seconds into the experiment.
    if trigger_wp.left_lane_marking.lane_change not in (
            carla.LaneChange.Left, carla.LaneChange.Both):
        raise ValueError('The fixed trigger does not permit a left lane change')

    source_wp = trigger_wp
    for _ in range(max(2, int(length))):
        source_wp = _single_waypoint(
            source_wp.next(1.0), 'lane-change corridor waypoint')
        destination_wp = source_wp.get_left_lane()
        if source_wp.is_junction:
            raise ValueError('The fixed lane-change trajectory enters the junction')
        if destination_wp is None or destination_wp.lane_type != carla.LaneType.Driving \
                or source_wp.lane_id * destination_wp.lane_id <= 0:
            raise ValueError('The fixed lane-change corridor has no valid left lane')


def build_cut_in_layout(carla_map, ego_anchor_location, config=None):
    """Build fixed spawn and trigger positions from the Town03 lane topology."""
    # [CUT_IN_SCENARIO] `trigger_waypoint` is computed once from the map. During
    # simulation the trigger does not reference ego position, speed, or policy.
    params = dict(CUT_IN_CONFIG)
    if config:
        params.update(config)

    ego_wp = carla_map.get_waypoint(
        ego_anchor_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if ego_wp is None or ego_wp.is_junction:
        raise ValueError('The ego anchor must project to a non-junction driving lane')

    right_wp = ego_wp.get_right_lane()
    if right_wp is None or right_wp.lane_type != carla.LaneType.Driving \
            or ego_wp.lane_id * right_wp.lane_id <= 0:
        raise ValueError('The cut-in scenario requires a same-direction right lane')

    front_wp = _single_waypoint(
        right_wp.next(params['front_gap']), 'front_sv spawn waypoint')
    rear_wp = _single_waypoint(
        right_wp.previous(params['rear_gap']), 'rear_sv spawn waypoint')
    trigger_wp = _single_waypoint(
        rear_wp.next(params['trigger_from_rear_spawn']), 'fixed cut-in trigger waypoint')

    if front_wp.is_junction or rear_wp.is_junction or trigger_wp.is_junction:
        raise ValueError('Spawn and trigger points must remain outside the junction')
    _validate_lane_change_corridor(trigger_wp, params['lane_change_length'])

    transforms = [_raised_transform(front_wp), _raised_transform(rear_wp)]
    agent_configs = [
        {
            'role': 'front_sv',
            'behavior': 'lane_follow',
            'initial_speed': params['front_speed'],
            'target_speed': params['front_speed'],
        },
        {
            'role': 'rear_sv',
            'behavior': 'fixed_point_cut_in',
            'initial_speed': params['rear_initial_speed'],
            'trigger_waypoint': trigger_wp,
            'overtake_speed': params['rear_overtake_speed'],
            'settled_speed': params['rear_settled_speed'],
            'lane_change_length': params['lane_change_length'],
        },
    ]

    return {
        'ego_transform': _raised_transform(ego_wp),
        'vehicle_transforms': transforms,
        'agent_configs': agent_configs,
        'trigger_transform': trigger_wp.transform,
        'config': params,
    }
