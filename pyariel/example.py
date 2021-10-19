from testsuite.rule_sets import rule_set
from testsuite.test_suite import test_suite, test_suite_scope
from pyariel.py_ariel import PyAriel


if __name__ == '__main__':
    ariel = PyAriel()
    ariel.run(rule_set, test_suite(), test_suite_scope())
