import ast
import random
import numpy as np

from autotest.auto_test import simulate_scenes
from autorepair.benchmarks.test_suite import TestSuite
from typing import List

class AutoTestSuite(TestSuite):
    def __init__(self, tests, metrics) -> None:
        self.tests = tests
        self.metrics = metrics

    def evaluate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        return self.run(rule_set, evaluation_tests_ids)

    def validate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        validation_scenes_ids = [test_id for test_id in self.tests if test_id not in evaluation_tests_ids]
        return self.run(rule_set, validation_scenes_ids)

    def run(self, rule_set: ast.Module, tests_ids: List[int]) -> dict:
        tests = [self.tests[test_id] for test_id in tests_ids]
        return simulate_scenes(rule_set, tests, self.metrics)
