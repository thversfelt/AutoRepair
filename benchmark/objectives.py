from typing import List

from highway_env.envs import AbstractEnv


def scores(env: AbstractEnv) -> List[float]:
    return [
        collision(env),
        speed_limit(env)
    ]


def collision(env: AbstractEnv) -> float:
    if env.vehicle.crashed:
        return -1.0
    else:
        return 0.0


def speed_limit(env: AbstractEnv) -> float:
    if env.vehicle.speed > env.config['speed_limit']:
        return -1.0
    else:
        return 0.0
