import ast
import unittest

from autotest.model.evaluation.instrumentation import Instrumenter


class TestInstrumentation(unittest.TestCase):
    def test_instrumenter(self):
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
        Instrumenter().visit(rule_set)
        ast.fix_missing_locations(rule_set)
        instrumented_source = ast.unparse(rule_set)
        
        expected_source = "\n".join([
            "def rule_set():",
            "   self.executed_statements = []",
            "   if one_var < 9:",
            "       self.executed_statements.append('one_var < 9')",
            "       return 1",
            "   else:",
            "       self.executed_statements.append('one_var < 9')",
            "       if other_var == 100:",
            "           self.executed_statements.append('other_var == 100')",
            "           return 2",
            "       else:",
            "           self.executed_statements.append('other_var == 100')",
            "           return 3"
        ])
        expected_rule_set = ast.parse(expected_source)
        ast.fix_missing_locations(expected_rule_set)
        expected_instrumented_source = ast.unparse(expected_rule_set)
        
        self.assertEqual(
            instrumented_source,
            expected_instrumented_source
        )