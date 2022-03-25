import ast
import inspect
import random
import autotest

from autotest.auto_test import AutoTest


if __name__ == '__main__':
    scene_ids = [96]
    #scene_ids = random.sample(range(0, 100), 10)
    auto_test = AutoTest()
    rule_set = ast.parse(inspect.getsource(autotest.model.modules.rule_set))
    results = auto_test.run(rule_set, scene_ids, visualized=True)
