# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import os
import autotest
import random
import matplotlib.pyplot as plt

from autotest.evaluation.metrics import AdaptiveCruiseControlMetric, TrafficLightManagementMetric
from autotest.autodrive import faulty_rule_set
from autorepair.ariel import Ariel
from autorepair.auto_repair import AutoRepair
from autorepair.testsuites.test_suites import AutoTestSuite
from l5kit.data import ChunkedDataset
from autotest.map import open_map
from autotest.scene import load_scenes


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset_path = root_path + "/autotest/autotest/data/sample/sample.zarr"
    dataset = ChunkedDataset(dataset_path).open()

    map_path = root_path + "/autotest/autotest/data/map/map.pkl"
    map = open_map(map_path)

    scenes_ids = random.sample(range(0, 100), 50)
    scenes = load_scenes(scenes_ids, dataset, map)
    

    metrics = [AdaptiveCruiseControlMetric, TrafficLightManagementMetric]
    faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))
    
    number_of_faults = 1
    budget = 180 # [s]

    test_suite = AutoTestSuite(scenes, metrics, parallelize=False)
    ariel_checkpoints = Ariel.repair(faulty_rule_set, test_suite, scenes_ids, budget)

    # The structure of the checkpoints is:
    #       checkpoints : {
    #           evaluation_time : { module : { 
    #               evaluation_results : {
    #                   scene_id : { metrics_scores : array([float, float]) },
    #                   ... (for each scene)
    #               },
    #               validation_results : { 
    #                   scene_id : { metrics_scores : array([float, float]) },
    #                   ... (for each scene)
    #               }
    #           },
    #           ... (for each evaluation time)
    #       }

    ariel_number_of_failures = {}

    for evaluation_time in ariel_checkpoints:
        module = list(ariel_checkpoints[evaluation_time].keys())[0]
        evaluation_results = ariel_checkpoints[evaluation_time][module]["evaluation_results"]
        failing_tests_ids = Ariel.failing_tests_ids(evaluation_results)
        number_of_failures = len(failing_tests_ids)
        ariel_number_of_failures[evaluation_time] = number_of_failures

    # Now we can plot the number of failures found over time
    plt.plot(list(ariel_number_of_failures.keys()), list(ariel_number_of_failures.values()), label="Ariel")
    plt.xlabel("Evaluation time [s]")
    plt.ylabel("Number of failures")
    plt.legend()
    plt.show()

    #last_checkpoint = list(checkpoints.keys())[-1]
    #solution = list(checkpoints[last_checkpoint].keys())[0]
    #print(ast.unparse(solution))
    print("Done!")