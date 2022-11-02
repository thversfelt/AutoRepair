# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import autotest
import random
import numpy as np
import matplotlib.pyplot as plt

from autotest.autodrive import faulty_rule_set
from autotest.auto_test import AutoTest
from autorepair.ariel import Ariel
from autorepair.test_suite import TestSuite
from typing import List


class AutoTestSuite(TestSuite):
    def __init__(self, simulator: AutoTest, evaluation_test_ids: List[int], validation_test_ids: List[int]):
        self.simulator = simulator
        self.evaluation_test_ids = evaluation_test_ids
        self.validation_test_ids = validation_test_ids

    def evaluate(self, rule_set: ast.Module) -> dict:
        return self.simulator.simulate(rule_set, self.evaluation_test_ids)
    
    def validate(self, rule_set: ast.Module) -> dict:
        return self.simulator.simulate(rule_set, self.validation_test_ids)

test_ids = list(range(0, 100))
simulator = AutoTest("sample", test_ids)
faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))

budget = 15
repetitions = 10
test_suite_cutoff = 50

# random_results = {}
# for repetition in range(repetitions):
#     evaluation_test_ids = random.sample(test_ids, len(test_ids))[:test_suite_cutoff]
#     validation_test_ids = [test_id for test_id in test_ids if test_id not in evaluation_test_ids]
#     test_suite = AutoTestSuite(simulator, evaluation_test_ids, validation_test_ids)
#     archives = Ariel.repair(faulty_rule_set, test_suite, budget, validate=True)
#     random_results[repetition] = archives

prioritized_results = {}
evaluation_test_ids = [
    88, 75, 46, 14, 87, 90,  6, 85,  9,  3, 35, 21, 48, 91, 86, 82, 79,
    22, 40, 77, 65, 54, 13, 39, 15, 59, 24,  8,  5, 60, 37, 34, 50, 70,
    33, 96,  7, 68, 45, 93, 27, 12, 89, 73, 58, 64, 83, 11, 74, 47, 97,
    95, 16, 23, 20, 66, 30, 94, 25, 56, 32, 71, 92, 38, 76, 49,  1, 42,
    52, 72, 44, 84, 78, 69, 29,  2, 31, 81,  4, 62, 61, 28, 41, 19, 57,
    67, 99, 53, 55, 26, 36, 17, 63, 10, 98, 18, 43, 51,  0, 80
]
for repetition in range(repetitions):
    validation_test_ids = [test_id for test_id in test_ids if test_id not in evaluation_test_ids]
    test_suite = AutoTestSuite(simulator, evaluation_test_ids, validation_test_ids)
    archives = Ariel.repair(faulty_rule_set, test_suite, budget, validate=True)
    prioritized_results[repetition] = archives

number_of_failing_tests_per_iteration = []
for iteration in range(budget):
    
    number_of_failing_tests_per_repetition = []
    for repetition in range(repetitions):
        archives = random_results[repetition]
        
        if iteration not in archives:
            number_of_failing_tests_per_repetition.append(0)
            continue
        
        archive = archives[iteration]
        
        evaluation_number_of_failing_tests = 0
        validation_number_of_failing_tests = 0
        
        for solution, results in archive.items():
            evaluation_scores = results["evaluation_results"]["metrics_scores"]
            validation_scores = results["validation_results"]["metrics_scores"]
            
            evaluation_failing_tests = [test_id for test_id, scores in evaluation_scores.items() if any(scores < 0)]
            validation_failing_tests = [test_id for test_id, scores in validation_scores.items() if any(scores < 0)]
            
            evaluation_number_of_failing_tests = len(evaluation_failing_tests) if len(evaluation_failing_tests) > evaluation_number_of_failing_tests else evaluation_number_of_failing_tests
            validation_number_of_failing_tests = len(validation_failing_tests) if len(validation_failing_tests) > validation_number_of_failing_tests else validation_number_of_failing_tests
            
        total_number_of_failing_tests = evaluation_number_of_failing_tests + validation_number_of_failing_tests
        number_of_failing_tests_per_repetition.append(total_number_of_failing_tests)
        
    number_of_failing_tests_per_iteration.append(number_of_failing_tests_per_repetition)
print(number_of_failing_tests_per_iteration)

average_number_of_failing_tests_per_iteration = np.average(number_of_failing_tests_per_iteration, axis=1)
print(average_number_of_failing_tests_per_iteration)

std_number_of_failing_tests_per_iteration = np.std(number_of_failing_tests_per_iteration, axis=1)
print(std_number_of_failing_tests_per_iteration)

fig, ax = plt.subplots()
ax.errorbar(x=list(range(budget)), y=average_number_of_failing_tests_per_iteration, yerr=std_number_of_failing_tests_per_iteration)
plt.ylim([0, 3])
plt.title('Number of failing tests per iteration (uniformly sampled test suite)')
plt.xlabel('Iteration')
plt.ylabel('Number of failing tests')

print("")
