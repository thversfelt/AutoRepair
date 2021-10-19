from typing import List
from highway_env.envs import AbstractEnv


def scores(env: AbstractEnv) -> List[float]:
    return [
        safety(env),
        speed(env)
    ]


def safety(env: AbstractEnv) -> float:
    # TODO: when a safe distance is passed to a near car, score is 0 or less, depending on how close to the nearest car
    # TODO: when the car crashed, score is -1
    # TODO: the further away from nearby cars, the more the score will be over 0
    if env.vehicle.crashed:
        return -1.0
    else:
        return 0.0


def speed(env: AbstractEnv) -> float:
    # TODO: when speed limit passed, score is 0 or less, depending on severity of how much over the limit
    # TODO: when under the speed limit, score is 0 or more, depending on how close to the speed limit
    if env.vehicle.speed > env.config['speed_limit']:
        return -1.0
    else:
        return 0.0
