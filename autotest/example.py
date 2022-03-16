import pandas as pd

from autotest.auto_test import AutoTest
from autotest.model.modules.rule_set import RuleSet


if __name__ == '__main__':
    scene_ids = [11]  # 96
    autotest = AutoTest()
    evaluation_results, instrumentation_results = autotest.run(RuleSet, scene_ids)
    print(pd.DataFrame.from_dict(evaluation_results))
