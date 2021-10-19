import gym
from testsuite.rule_sets import correct_rule_set
from testsuite.utilities import simulate_env
from testsuite.envs import *

if __name__ == '__main__':
    env = gym.make('rear-env-v0')
    simulate_env(env, correct_rule_set, render=True)
