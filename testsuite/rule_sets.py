from highway_env.envs import AbstractEnv
from testsuite.features import adaptive_cruise_control, emergency_stop
from testsuite.utilities import get_front_gap


def rule_set(env: AbstractEnv):
    if get_front_gap(env.vehicle) + 4.0 * env.vehicle.speed < 0:
        acceleration = adaptive_cruise_control(env)
    else:
        acceleration = emergency_stop(env)
    return [acceleration, 0.0]


def correct_rule_set(env: AbstractEnv):
    if get_front_gap(env.vehicle) - 4.0 * env.vehicle.speed > 0:
        acceleration = adaptive_cruise_control(env)
    else:
        acceleration = emergency_stop(env)
    return [acceleration, 0.0]