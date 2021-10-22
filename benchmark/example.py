import gym
from benchmark.rule_sets import correct_rule_set
from benchmark.utilities import simulate_env
from benchmark.envs import *

if __name__ == '__main__':
    env = gym.make('rear-env-v0')
    simulate_env(env, correct_rule_set, render=True)
