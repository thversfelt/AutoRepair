import ast
import inspect
import random
import autotest

from autotest.auto_test import AutoTest


if __name__ == '__main__':
    #scene_ids = [6, 11, 12, 13, 20, 22, 25, 26, 32, 54]
    #scene_ids = list(range(52, 55))
    scene_ids = random.sample(range(0, 100), 10)
    #scene_ids = list(range(0, 100))
    #scene_ids = [90]
    print(scene_ids)
    
    auto_test = AutoTest()
    rule_set = ast.parse(inspect.getsource(autotest.model.modules.rule_set))
    results = auto_test.run(rule_set, scene_ids, visualized=True)
    print(results)
