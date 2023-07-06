# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import autotest
import random


from autotest.evaluation.metrics import AdaptiveCruiseControlMetric, TrafficLightManagementMetric
from autotest.autodrive import faulty_rule_set
from autorepair.ariel import Ariel
from autorepair.auto_repair import AutoRepair
from autorepair.testsuites.test_suites import AutoTestSuite
from l5kit.data import ChunkedDataset
from autotest.map import open_map
from autotest.scene import load_scenes

if __name__ == '__main__':
    dataset_path = "../autotest/autotest/data/validate/validate.zarr"
    map_path = "../autotest/autotest/data/map/map.pb"
    loaded_map_path = "../autotest/autotest/data/map/map.pkl"

    scenes_ids = random.sample(range(0, 15631), 50)
    dataset = ChunkedDataset(dataset_path).open()
    map = open_map(loaded_map_path) # OF load_map(map_path) ALS de map nog niet is opgeslagen
    scenes = load_scenes(scenes_ids, dataset, map)
    
    #scenes_features = featurize_scenes(scenes)
    #prioritized_scenes_ids = prioritize_scenes(scenes_features)
    
    metrics = [AdaptiveCruiseControlMetric, TrafficLightManagementMetric]
    
    faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))
    
    number_of_faults = 1
    budget = 30 # [s]

    test_suite = AutoTestSuite(scenes, metrics, parallelize=False)

    ariel_checkpoints = Ariel.repair(faulty_rule_set, test_suite, scenes_ids, budget)
    auto_repair_checkpoints = AutoRepair.repair(faulty_rule_set, test_suite, scenes_ids, budget)

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
    auto_repair_number_of_failures = {}

    for evaluation_time in ariel_checkpoints:
        module = list(ariel_checkpoints[evaluation_time].keys())[0]
        evaluation_results = ariel_checkpoints[evaluation_time][module]["evaluation_results"]
        failing_tests_ids = Ariel.failing_tests_ids(evaluation_results)
        number_of_failures = len(failing_tests_ids)
        ariel_number_of_failures[evaluation_time] = number_of_failures
    
    for evaluation_time in auto_repair_checkpoints:
        module = list(auto_repair_checkpoints[evaluation_time].keys())[0]
        evaluation_results = auto_repair_checkpoints[evaluation_time][module]["evaluation_results"]
        failing_tests_ids = AutoRepair.failing_tests_ids(evaluation_results)
        number_of_failures = len(failing_tests_ids)
        auto_repair_number_of_failures[evaluation_time] = number_of_failures

    # Now we can compare the checkpoints of Ariel and AutoRepair, by plotting the number of failures found over time.
    import matplotlib.pyplot as plt

    plt.plot(list(ariel_number_of_failures.keys()), list(ariel_number_of_failures.values()), label="Ariel")
    plt.plot(list(auto_repair_number_of_failures.keys()), list(auto_repair_number_of_failures.values()), label="AutoRepair")
    plt.xlabel("Evaluation time [s]")
    plt.ylabel("Number of failures")
    plt.legend()
    plt.show()

        

    #last_checkpoint = list(checkpoints.keys())[-1]
    #solution = list(checkpoints[last_checkpoint].keys())[0]
    
    #print(ast.unparse(solution))
    #print(checkpoints)
    print("Done!")