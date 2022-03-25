import ast
import inspect

from autotest.auto_test import AutoTest
from autotest.model.modules.rule_set import RuleSet


if __name__ == '__main__':
    scene_ids = [96]
    autotest = AutoTest()
    rule_set = ast.parse(inspect.getsource(RuleSet))
    results = autotest.run(rule_set, scene_ids)
