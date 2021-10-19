from typing import Dict, List, Callable
import gym
from highway_env.envs import AbstractEnv
from testsuite.features import *
from testsuite.utilities import *
from testsuite.envs import *


def test_suite() -> List[Callable]:
    return [
        test_follow_env,
        test_jam_env,
        test_rear_env
    ]


def test_suite_scope() -> Dict:
    return globals()


def test_follow_env(rule_set: Callable) -> List[float]:
    env = gym.make('follow-env-v0')
    return simulate_env(env, rule_set)


def test_jam_env(rule_set: Callable) -> List[float]:
    env = gym.make('jam-env-v0')
    return simulate_env(env, rule_set)


def test_rear_env(rule_set: Callable) -> List[float]:
    env = gym.make('rear-env-v0')
    return simulate_env(env, rule_set)
