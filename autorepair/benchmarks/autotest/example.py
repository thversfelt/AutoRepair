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
from featurizer import featurize_scenes
from prioritizer import prioritize_scenes
from autorepair.test_suite import TestSuite
from typing import List
from tqdm import tqdm


    
class RandomAutoTestSuite(TestSuite):
    def __init__(self, scenes, cutoff, metrics) -> None:
        self.scenes = scenes
        self.cutoff = cutoff
        self.metrics = metrics
        
        # Randomize the order of the scenes.
        self.random_scenes_ids = list(scenes.keys())
        random.shuffle(self.random_scenes_ids)
        
    def evaluate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        # Evaluate the scenes up to the cutoff.
        evaluation_scenes = [self.scenes[scene_id] for scene_id in self.random_scenes_ids[:self.cutoff]]
        return simulate_scenes(rule_set, evaluation_scenes, self.metrics)
    
    def validate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        # Validate the scenes after the cutoff.
        validation_scenes = [self.scenes[scene_id] for scene_id in self.random_scenes_ids[self.cutoff:]]
        return simulate_scenes(rule_set, validation_scenes, self.metrics)

class PrioritizedAutoTestSuite(TestSuite):
    def __init__(self, scenes, cutoff, metrics) -> None:
        self.scenes = scenes
        self.cutoff = cutoff
        self.metrics = metrics
        
        scenes_features = featurize_scenes(self.scenes)
        self.prioritized_scenes_ids = prioritize_scenes(scenes_features)
        
    def evaluate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        # Evaluate the scenes with the highest priority, up to the cutoff.
        evaluation_scenes = [self.scenes[scene_id] for scene_id in self.prioritized_scenes_ids[:self.cutoff]]
        return simulate_scenes(rule_set, evaluation_scenes, self.metrics)
    
    def validate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        # Validate the scenes with the lowest priority, after the cutoff.
        validation_scenes = [self.scenes[scene_id] for scene_id in self.prioritized_scenes_ids[self.cutoff:]]
        return simulate_scenes(rule_set, validation_scenes, self.metrics)

class FailingAutoTestSuite(TestSuite):
    def __init__(self, scenes, metrics) -> None:
        self.scenes = scenes
        self.metrics = metrics
    
    def evaluate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        if parent_evaluation_results is None:
            evaluation_scenes = list(self.scenes.values())
            return simulate_scenes(rule_set, evaluation_scenes, self.metrics)
        else:
            evaluation_scenes = [self.scenes[scene_id] for scene_id in self.failing_scenes_ids(parent_evaluation_results)]
            return simulate_scenes(rule_set, evaluation_scenes, self.metrics)
    
    def validate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        if parent_evaluation_results is None:
            return None
        else:
            failing_scenes_ids = self.failing_scenes_ids(parent_evaluation_results)
            validation_scenes = [self.scenes[scene_id] for scene_id in self.scenes.keys() if scene_id not in failing_scenes_ids]
            return simulate_scenes(rule_set, validation_scenes, self.metrics)
    
    def failing_scenes_ids(self, parent_evaluation_results: dict = None) -> List[int]:
        failing_scenes_ids = []
        for scene_id, scene_evaluation_results in parent_evaluation_results.items():
            metrics_scores = scene_evaluation_results["metrics_scores"]
            if np.any(metrics_scores < 0):
                failing_scenes_ids.append(scene_id)
        return failing_scenes_ids

if __name__ == '__main__':
    dataset_name = "validate"
    scenes_ids = [3093, 4606, 7007, 492, 15835, 2522, 7266, 4129, 564, 5606, 13945, 4607, 1798, 1746, 15055, 5692, 6850, 
                  12422, 15955, 3363, 1428, 12703, 6784, 12886, 10420, 11189, 1882, 3710, 578, 9925, 14838, 4469, 4772, 
                  2288, 9165, 13955, 11894, 12448, 15077, 3013, 154, 10328, 4041, 5833, 10787, 6967, 5509, 2110, 6314, 
                  11512, 4932, 3108, 1899, 12652, 11650, 2818, 2271, 6146, 9945, 1727, 1507, 2309, 1236, 9475, 13510, 
                  3479, 3058, 4412, 14869, 11153, 10085, 2772, 9086, 14435, 6215, 997, 10753, 15039, 15341, 6182, 10104, 
                  983, 85, 2718, 3288, 429, 10121, 13506, 762, 7360, 7477, 5725, 2820, 624, 9084, 1514, 9064, 10774, 
                  15662, 2645]
    scenes = load_scenes(scenes_ids, dataset_name)
    metrics = [AdaptiveCruiseControlMetric(), TrafficLightManagementMetric()]
    faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))
   
    budget = 10
    number_of_faults = 3
    repetitions = 20

    # Create a progress bar to show the progress of the experiments.
    progress_bar = tqdm(desc=f'Running experiment', total=repetitions)
    
    results = {}
    for repetition in range(repetitions):
        test_suite = FailingAutoTestSuite(scenes, metrics)
        archives = Ariel.repair(faulty_rule_set, test_suite, budget, validate=True)
        results[repetition] = archives
        progress_bar.update(n=1)
        
    # Save the results to a file.
    with open(f"failing_test_suite_number_of_faults_{number_of_faults}.pkl", "wb") as file:
        pickle.dump(results, file)

    # test_suite_cutoffs = [50, 75]

    # for test_suite_cutoff in test_suite_cutoffs:
    #     # Create a progress bar to show the progress of the experiments.
    #     progress_bar = tqdm(desc=f'Running experiment with test suite cutoff {test_suite_cutoff}', total=repetitions)
        
    #     results = {}
    #     for repetition in range(repetitions):
    #         # Suffle the test ids every repetition.
    #         random.shuffle(random_scenes_ids)
    #         evaluation_test_ids = random_scenes_ids[:test_suite_cutoff]
    #         validation_test_ids = random_scenes_ids[test_suite_cutoff:]
    #         test_suite = AutoTestSuite(scenes, evaluation_test_ids, validation_test_ids)
            
    #         archives = Ariel.repair(faulty_rule_set, test_suite, budget, validate=True)
    #         results[repetition] = archives
    #         progress_bar.update(n=1)
            
    #     # Save the results to a file.
    #     with open(f"random_test_suite_cutoff_{test_suite_cutoff}_number_of_faults_{number_of_faults}.pkl", "wb") as file:
    #         pickle.dump(results, file)