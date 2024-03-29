import ast

from autotest.auto_test import simulate_scenes
from typing import List


class TestSuite:
    def __init__(self) -> None:
        pass
    
    def evaluate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        pass
    
    def validate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        pass
    
    def run(self, rule_set: ast.Module, tests_ids: List[int]) -> dict:
        pass

class AutoTestSuite(TestSuite):
    def __init__(self, tests, metrics, parallelize, verbose=True) -> None:
        self.tests = tests
        self.metrics = metrics
        self.parallelize = parallelize
        self.verbose = verbose

    def evaluate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        return self.run(rule_set, evaluation_tests_ids)

    def validate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        validation_tests_ids = [test_id for test_id in self.tests if test_id not in evaluation_tests_ids]
        return self.run(rule_set, validation_tests_ids)

    def run(self, rule_set: ast.Module, tests_ids: List[int]) -> dict:
        tests = [self.tests[test_id] for test_id in tests_ids]
        return simulate_scenes(rule_set, tests, self.metrics, parallelize=self.parallelize, verbose=self.verbose)