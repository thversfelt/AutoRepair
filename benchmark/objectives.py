from typing import List
from highway_env.envs import AbstractEnv
import numpy as np
from benchmark import utilities


def scores(env: AbstractEnv) -> List[float]:
    return [
        safety(env),
        speed(env)
    ]


def safety(env: AbstractEnv) -> float:
    # When a safe distance is passed to a near car, score is 0 or less, depending on how close to the nearest car
    # When the car crashed, score is -1
    # The further away from nearby cars, the more the score will be over 0
    controlled_vehicle = env.vehicle
    min_distance = None
    for other_vehicle in env.road.vehicles:
        if other_vehicle is controlled_vehicle:
            continue
        distance = np.linalg.norm(other_vehicle.position - controlled_vehicle.position)
        if min_distance is None:
            min_distance = distance
        elif distance < min_distance:
            min_distance = distance

    safe_distance = 2.0 * controlled_vehicle.LENGTH
    preferred_distance = 4.0 * controlled_vehicle.speed

    if min_distance >= safe_distance:
        return utilities.clamp(min_distance / preferred_distance, 0.0, 1.0)
    else:
        return utilities.clamp(-1.0 + min_distance / safe_distance, -1.0, 0.0)


def speed(env: AbstractEnv) -> float:
    # When speed limit passed, score is 0 or less, depending on severity of how much over the limit
    # When under the speed limit, score is 0 or more, depending on how close to the speed limit
    controlled_vehicle = env.vehicle
    speed_limit = env.config['speed_limit']
    if controlled_vehicle.speed <= speed_limit:
        return utilities.clamp(controlled_vehicle.speed / speed_limit, 0.0, 1.0)
    else:
        return utilities.clamp((speed_limit - controlled_vehicle.speed) / speed_limit, -1.0, 0.0)
