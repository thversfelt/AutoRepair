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
        statement = 'other_var == 100'
        
        path_references, statement_reference = find_statements_references(rule_set, path, statement)
        path_references_as_strings = [ast.unparse(reference.test) for reference in path_references]
        statement_reference_as_string = ast.unparse(statement_reference.test)
        
        self.assertEqual(
            set(path),
            set(path_references_as_strings)
        )
        self.assertEqual(
            statement,
            statement_reference_as_string
        )