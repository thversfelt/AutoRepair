import ast
import inspect
from autotest.auto_test import AutoTest
from autotest.model.modules.rule_set import RuleSet
from pyariel.py_ariel import PyAriel


if __name__ == '__main__':
    scene_ids = [11]  # 96
    autotest = AutoTest()
    rule_set = ast.parse(inspect.getsource(RuleSet))
    evaluation_results, instrumentation_results = autotest.run(rule_set, scene_ids)
    
    ariel = PyAriel()
    ariel.run(autotest, rule_set, scene_ids)
