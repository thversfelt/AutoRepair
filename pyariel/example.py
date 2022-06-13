import ast
import inspect
import autotest

from autotest.auto_test import AutoTest
from pyariel.py_ariel import PyAriel

if __name__ == '__main__':
    
    rule_set = ast.parse(inspect.getsource(autotest.model.modules.rule_set))
    auto_test = AutoTest()
    #scene_ids = [96]  # 96
    scene_ids = [11, 12, 13, 20, 26, 32, 54]
    
    ariel = PyAriel()
    ariel.run(rule_set, auto_test, scene_ids)
