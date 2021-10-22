from highway_env.envs import AbstractEnv
from benchmark.features import *
from benchmark.utilities import *


def rule_set(env: AbstractEnv):
    if get_rear_vehicle_gap(env.vehicle) * (get_rear_vehicle_speed(env.vehicle) - env.vehicle.speed) < 1.0:
        acceleration = automated_emergency_acceleration(env)
    else:
        if get_front_vehicle_gap(env.vehicle) + 4.0 * env.vehicle.speed < 0:
            acceleration = adaptive_cruise_control(env)
        else:
            acceleration = automated_emergency_braking(env)
    return [acceleration, 0.0]


def correct_rule_set(env: AbstractEnv):
    if get_rear_vehicle_gap(env.vehicle) / (get_rear_vehicle_speed(env.vehicle) - env.vehicle.speed) < 4.0:
        acceleration = automated_emergency_acceleration(env)
    else:
        if get_front_vehicle_gap(env.vehicle) - 4.0 * env.vehicle.speed > 0:
            acceleration = adaptive_cruise_control(env)
        else:
            acceleration = automated_emergency_braking(env)
    return [acceleration, 0.0]