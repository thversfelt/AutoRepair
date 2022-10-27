# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import autotest
import random

from autotest.autodrive import faulty_rule_set
from autotest.auto_test import AutoTest
from autorepair.ariel import Ariel
from autorepair.test_suite import TestSuite
from typing import List


class ExampleTestSuite(TestSuite):
    def __init__(self, simulator: AutoTest, test_ids: List[int]):
        self.test_ids = test_ids
        self.simulator = simulator

    def evaluate(self, rule_set: ast.Module) -> dict:
        return self.simulator.simulate(rule_set, self.test_ids)

all_test_ids = list(range(0, 100))
simulator = AutoTest("sample", all_test_ids)
faulty_rule_set = ast.parse(inspect.getsource(autotest.autodrive.faulty_rule_set))

budget = 15
repetitions = 20
test_suite_cutoff = 50

random_results = {}
random_test_ids = random.sample(all_test_ids, len(all_test_ids)) #[55, 85, 68]
for repetition in range(repetitions):
    test_suite = ExampleTestSuite(simulator, random_test_ids[:test_suite_cutoff])
    archives = Ariel.repair(faulty_rule_set, test_suite, budget)
    random_results[repetition] = archives

prioritized_results = {}
prioritized_test_ids = [
    88, 75, 46, 14, 87, 90,  6, 85,  9,  3, 35, 21, 48, 91, 86, 82, 79,
    22, 40, 77, 65, 54, 13, 39, 15, 59, 24,  8,  5, 60, 37, 34, 50, 70,
    33, 96,  7, 68, 45, 93, 27, 12, 89, 73, 58, 64, 83, 11, 74, 47, 97,
    95, 16, 23, 20, 66, 30, 94, 25, 56, 32, 71, 92, 38, 76, 49,  1, 42,
    52, 72, 44, 84, 78, 69, 29,  2, 31, 81,  4, 62, 61, 28, 41, 19, 57,
    67, 99, 53, 55, 26, 36, 17, 63, 10, 98, 18, 43, 51,  0, 80
]
for repetition in range(repetitions):
    test_suite = ExampleTestSuite(simulator, prioritized_test_ids[:test_suite_cutoff])
    archives = Ariel.repair(faulty_rule_set, test_suite, budget)
    prioritized_results[repetition] = archives

number_of_failing_tests_per_iteration_prioritized = []
for iteration in range(budget):
    number_of_failing_tests_per_repetition = []
    for repetition in range(repetitions):
        archives = random_results[repetition]
        if iteration not in archives:
            continue
        archive = archives[iteration]
        number_of_failing_tests = 99999
        for solution, results in archive.items():
            metrics_scores = results["metrics_scores"]
            failing_tests = [test_id for test_id, scores in metrics_scores.items() if any(scores < 0)]
            number_of_failing_tests = len(failing_tests) if len(failing_tests) < number_of_failing_tests else number_of_failing_tests
        number_of_failing_tests_per_repetition.append(number_of_failing_tests)
    number_of_failing_tests_per_iteration_prioritized.append(number_of_failing_tests_per_repetition)
print(number_of_failing_tests_per_iteration_prioritized)



print("")
