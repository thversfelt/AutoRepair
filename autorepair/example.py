# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import autotest
import random


from autotest.evaluation.metrics import AdaptiveCruiseControlMetric, TrafficLightManagementMetric
from autotest.autodrive import faulty_rule_set
from autorepair.ariel import Ariel
from autorepair.testsuites.autotest.featurizer import featurize_scenes
from autorepair.testsuites.autotest.prioritizer import prioritize_scenes
from autorepair.testsuites.test_suites import AutoTestSuite
from l5kit.data import ChunkedDataset
from autotest.map import open_map
from autotest.scene import load_scenes

if __name__ == '__main__':
    dataset_path = "../autotest/autotest/data/validate/validate.zarr"
    map_path = "../autotest/autotest/data/map/map.pb"
    loaded_map_path = "../autotest/autotest/data/map/map.pkl"

    scenes_ids = random.sample(range(0, 15631), 100)
    dataset = ChunkedDataset(dataset_path).open()
    map = open_map(loaded_map_path) # OF load_map(map_path) ALS de map nog niet is opgeslagen
    scenes = load_scenes(scenes_ids, dataset, map, parallelize=True)
    
    #scenes_features = featurize_scenes(scenes)
    #prioritized_scenes_ids = prioritize_scenes(scenes_features)
    
    metrics = [AdaptiveCruiseControlMetric, TrafficLightManagementMetric]
    
    faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))
    
    number_of_faults = 1
    budget = 180 # [s]

    test_suite = AutoTestSuite(scenes, metrics, parallelize=False)
    checkpoints = Ariel.repair(faulty_rule_set, test_suite, scenes_ids, budget)
    last_checkpoint = list(checkpoints.keys())[-1]
    solution = list(checkpoints[last_checkpoint].keys())[0]
    
    print(ast.unparse(solution))
    print(checkpoints)
    print("Done!")