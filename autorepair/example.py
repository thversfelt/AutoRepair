# If imports are not working: https://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages
import ast
import inspect
import autotest
import random

from autotest.autodrive import rule_set
from autotest.auto_test import AutoTest
import numpy as np
from autorepair.ariel import Ariel
from autorepair.mutations import shift
from autorepair.test_suite import TestSuite
from typing import List


class ExampleTestSuite(TestSuite):
    def __init__(self, simulator: AutoTest, test_ids: List[int]):
        self.test_ids = test_ids
        self.simulator = simulator

    def evaluate(self, rule_set: ast.Module) -> dict:
        return self.simulator.simulate(rule_set, self.test_ids, visualize=True)

test_ids = [96]
simulator = AutoTest("sample", test_ids)

rule_set = ast.parse(inspect.getsource(autotest.autodrive.rule_set))
test_suite = ExampleTestSuite(simulator, test_ids)

patch = Ariel.repair(rule_set, test_suite)


# rule_set = ast.parse(inspect.getsource(rule_set))
# path = ["traffic_light.color == 'RED'", "traffic_light.color == 'YELLOW'", "traffic_light == None"]
# statement = "traffic_light == None"
# modify(rule_set, path, statement)
