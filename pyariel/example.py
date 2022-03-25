import ast
import inspect
import autotest

from autotest.auto_test import AutoTest
from pyariel.py_ariel import PyAriel

if __name__ == '__main__':
    
    auto_test = AutoTest()
    rule_set = ast.parse(inspect.getsource(autotest.model.modules.rule_set))
    scene_ids = [11]  # 96
    
    ariel = PyAriel()
    ariel.run(rule_set, auto_test, scene_ids)
