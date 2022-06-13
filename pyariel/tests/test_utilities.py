import ast
import unittest

from pyariel.utilities import find_statements_references


class TestUtilities(unittest.TestCase):
    def test_reference_finder(self):
        source = "\n".join([
            "def rule_set():",
            "   if one_var < 9:",
            "       return 1",
            "   else:",
            "       if other_var == 100:",
            "           return 2",
            "       else:",
            "           return 3"
        ])
        
        rule_set = ast.parse(source)
        path = ['one_var < 9', 'other_var == 100']
        
        references = find_statements_references(rule_set, path)
        expected_references = [ast.unparse(reference.test) for reference in references.values()]
        
        self.assertEqual(
            set(path),
            set(expected_references)
        )