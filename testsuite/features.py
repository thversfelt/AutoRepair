from highway_env.envs import AbstractEnv
from testsuite.utilities import get_front_gap, get_front_vehicle


IDLE_ACCELERATION: float = 0.0  # m/s^2
ACC_ACCELERATION: float = 0.5  # m/s^2
ACC_DECELERATION: float = -0.5  # m/s^2
ES_DECELERATION: float = -1.0  # m/s^2


def adaptive_cruise_control(env: AbstractEnv) -> float:  # longitudinal control
    leading_vehicle = get_front_vehicle(env.vehicle)
    if leading_vehicle is None:
        return IDLE_ACCELERATION

    if leading_vehicle.speed > env.vehicle.speed:
        return ACC_ACCELERATION
    else:
        return ACC_DECELERATION


def emergency_stop(env: AbstractEnv) -> float:  # longitudinal control
    if get_front_vehicle(env.vehicle) is None:
        return IDLE_ACCELERATION
    else:
        return ES_DECELERATION
