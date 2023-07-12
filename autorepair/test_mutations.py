import unittest
import ast
from autorepair.utilities import find_dominated_statements, find_dominating_statements, find_statement_reference
from mutations import modify_threshold_value, shift

class TestMutations(unittest.TestCase):
    
    def test_modify_threshold_value(self):
        # Test the modify_threshold_value function
        threshold = ast.parse("threshold = 10").body[0].value
        old_value = threshold.value
        modify_threshold_value(threshold)
        new_value = threshold.value
        self.assertNotEqual(old_value, new_value)
        
    def test_shift(self):
        # Test the shift function
        individual = ast.parse(
            "if x > 0:\n" +
            "    z = 1\n" +
            "else:\n" +
            "    if y > 0:\n" +
            "        z = 2\n" +
            "    else:\n" +
            "        z = 3"
        ).body[0]

        path = ["x > 0", "y > 0"]
        statement = "x > 0"

        old_individual = ast.unparse(individual)
        shift(individual, path, statement)
        new_individual = ast.unparse(individual)
        self.assertNotEqual(old_individual, new_individual)

if __name__ == '__main__':
    unittest.main()