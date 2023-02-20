# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import autotest
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle


from autotest.auto_test import simulate_scenes
from autotest.scenes_loader import load_scenes
from autotest.evaluation.metrics import AdaptiveCruiseControlMetric, TrafficLightManagementMetric
from autotest.autodrive import faulty_rule_set
from autorepair.ariel import Ariel
from autorepair.testsuites.test_suites import TestSuite, AutoTestSuite
from featurizer import featurize_scenes
from prioritizer import prioritize_scenes
from typing import List
from tqdm import tqdm


if __name__ == '__main__':
    dataset_name = "validate"
    #scenes_ids = random.sample(range(0, 16220), 1000)
    scenes_ids = [3093, 4606, 7007, 492, 15835, 2522, 7266, 4129, 564, 5606, 13945, 4607, 1798, 1746, 15055, 5692, 6850, 
                  12422, 15955, 3363, 1428, 12703, 6784, 12886, 10420, 11189, 1882, 3710, 578, 9925, 14838, 4469, 4772, 
                  2288, 9165, 13955, 11894, 12448, 15077, 3013, 154, 10328, 4041, 5833, 10787, 6967, 5509, 2110, 6314, 
                  11512, 4932, 3108, 1899, 12652, 11650, 2818, 2271, 6146, 9945, 1727, 1507, 2309, 1236, 9475, 13510, 
                  3479, 3058, 4412, 14869, 11153, 10085, 2772, 9086, 14435, 6215, 997, 10753, 15039, 15341, 6182, 10104, 
                  983, 85, 2718, 3288, 429, 10121, 13506, 762, 7360, 7477, 5725, 2820, 624, 9084, 1514, 9064, 10774, 
                  15662, 2645]
    scenes = load_scenes(scenes_ids, dataset_name)
    #scenes_features = featurize_scenes(scenes)
    #prioritized_scenes_ids = prioritize_scenes(scenes_features)
    metrics = [AdaptiveCruiseControlMetric(), TrafficLightManagementMetric()]
    
    faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))
    number_of_faults = 2
    
    cutoff = 50
    budget = 100 # [s]

    test_suite = AutoTestSuite(scenes, metrics)
    evaluation_tests_ids = scenes_ids[:cutoff]
    checkpoints = Ariel.repair(faulty_rule_set, test_suite, evaluation_tests_ids, budget, validate=True, evaluate_failing_tests=True)
    last_checkpoint = list(checkpoints.keys())[-1]
    solution = list(checkpoints[last_checkpoint].keys())[0]
    
    print(ast.unparse(solution))
    print(checkpoints)