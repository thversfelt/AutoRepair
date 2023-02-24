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
from autorepair.testsuites.autotest.featurizer import featurize_scenes
from autorepair.testsuites.autotest.prioritizer import prioritize_scenes
from autorepair.testsuites.test_suites import TestSuite, AutoTestSuite
from typing import List
from tqdm import tqdm


if __name__ == '__main__':
    dataset_name = "sample"
    scenes_ids = [0, 1, 2, 3, 4]
    scenes = load_scenes(scenes_ids, dataset_name)
    scenes_features = featurize_scenes(scenes)
    prioritized_scenes_ids = prioritize_scenes(scenes_features)
    
    metrics = [AdaptiveCruiseControlMetric(), TrafficLightManagementMetric()]
    
    faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))
    number_of_faults = 2

    budget = 100 # [s]

    test_suite = AutoTestSuite(scenes, metrics)
    checkpoints = Ariel.repair(faulty_rule_set, test_suite, scenes_ids, budget)
    last_checkpoint = list(checkpoints.keys())[-1]
    solution = list(checkpoints[last_checkpoint].keys())[0]
    
    print(ast.unparse(solution))
    print(checkpoints)