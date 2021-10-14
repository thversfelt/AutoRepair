from typing import Callable, Dict, List
import gym
from benchmark.envs import *
from benchmark.objectives import *
from benchmark.features import *
from benchmark.utilities import *


def test_suite() -> List[Callable]:
    return [
        test_follow_env,
        test_jam_env
    ]


def test_suite_scope() -> Dict:
    return globals()


def test_follow_env(rule_set: Callable) -> List[float]:
    env = gym.make('follow-env-v0')
    return test_env(env, rule_set)


def test_jam_env(rule_set: Callable) -> List[float]:
    env = gym.make('jam-env-v0')
    return test_env(env, rule_set)


def test_env(env: AbstractEnv, rule_set: Callable) -> List[float]:
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
    return min_score
