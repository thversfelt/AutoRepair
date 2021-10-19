from highway_env.envs import AbstractEnv
from testsuite.utilities import get_front_vehicle


IDLE_ACCELERATION: float = 0.0  # Idle acceleration (constant speed) [m/s^2]
ACC_ACCELERATION: float = 0.5  # Adaptive Cruise Control (ACC) acceleration [m/s^2]
ACC_DECELERATION: float = -0.5  # Adaptive Cruise Control (ACC) braking [m/s^2]
AEB_DECELERATION: float = -1.0  # Automated Emergency Braking (AEB) [m/s^2]
AEA_ACCELERATION: float = 1.0  # Automated Emergency Acceleration (AEA) [m/s^2]


def adaptive_cruise_control(env: AbstractEnv) -> float:  # longitudinal control
    leading_vehicle = get_front_vehicle(env.vehicle)
    if leading_vehicle is None:
        return IDLE_ACCELERATION

    if leading_vehicle.speed > env.vehicle.speed:
        return ACC_ACCELERATION
    else:
        return ACC_DECELERATION


def automated_emergency_braking(env: AbstractEnv) -> float:  # longitudinal control
    return AEB_DECELERATION


def automated_emergency_acceleration(env: AbstractEnv) -> float:  # longitudinal control
    return AEA_ACCELERATION
