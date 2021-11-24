import sys
from typing import Callable, List
import numpy as np
from highway_env.envs import AbstractEnv
from highway_env.vehicle.kinematics import Vehicle


def get_rear_vehicle(vehicle: Vehicle) -> Vehicle:
    _, rear_vehicle = vehicle.road.neighbour_vehicles(vehicle)
    return rear_vehicle


def get_rear_vehicle_gap(vehicle: Vehicle) -> float:
    rear_vehicle = get_rear_vehicle(vehicle)
    if rear_vehicle is None:
        return sys.maxsize
    else:
        return np.linalg.norm(rear_vehicle.position - vehicle.position)


def get_rear_vehicle_speed(vehicle: Vehicle) -> float:
    rear_vehicle = get_rear_vehicle(vehicle)
    if rear_vehicle is None:
        return 0.0000000000000001
    else:
        return rear_vehicle.speed


def get_front_vehicle(vehicle: Vehicle) -> Vehicle:
    front_vehicle, _ = vehicle.road.neighbour_vehicles(vehicle)
    return front_vehicle


def get_front_vehicle_gap(vehicle: Vehicle) -> float:
    front_vehicle = get_front_vehicle(vehicle)
    if front_vehicle is None:
        return sys.maxsize
    else:
        return np.linalg.norm(front_vehicle.position - vehicle.position)


def simulate_env(env: AbstractEnv, rule_set: Callable, render: bool = False) -> List[float]:
    env.reset()
    min_score = None
    done = False
    while not done:
        action = rule_set(env)
        state, score, done, _ = env.step(action)
        if min_score is None:
            min_score = score
        else:
            min_score = np.minimum(min_score, score)

        if render:
            env.render()
    return min_score


def clamp(x: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(x, maximum))
