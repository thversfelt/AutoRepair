from benchmark.features import rule_set
from benchmark.test_suite import test_suite, test_suite_scope
from py_ariel.ariel import Ariel


if __name__ == '__main__':
    ariel = Ariel()
    ariel.run(rule_set, test_suite(), test_suite_scope())
