import pandas as pd

from autotest.auto_test import AutoTest


if __name__ == '__main__':
    scene_ids = [11]  # 96
    autotest = AutoTest()
    results = autotest.run(scene_ids, visualized=True)
    print(pd.DataFrame.from_dict(results))
