import unittest
import ast
import astor

from pyariel import utilities, mutations


class TestMutations(unittest.TestCase):
    def test_swap_neighbor_dominator(self):
        actual_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 5:",
            "           return 6",
            "       else:",
            "           if 8:",
            "               return 9",
            "           else:",
            "               if 11:",
            "                   return 12",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        actual_rule_set = ast.parse(actual_source)

        path_lines = [2, 5, 8]
        one_statement_line = 8
        other_statement_line = 5

        path, one_statement = utilities.find_path_references(actual_rule_set, path_lines, one_statement_line)
        path, other_statement = utilities.find_path_references(actual_rule_set, path_lines, other_statement_line)
        mutations.swap(actual_rule_set, path, one_statement, other_statement)
        ast.fix_missing_locations(actual_rule_set)

        expected_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 8:",
            "           return 9",
            "       else:",
            "           if 5:",
            "               return 6",
            "           else:",
            "               if 11:",
            "                   return 12",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        expected_rule_set = ast.parse(expected_source)

        self.assertEqual(
            astor.to_source(expected_rule_set),
            astor.to_source(actual_rule_set)
        )

    def test_swap_non_neighbor_dominator(self):
        actual_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 5:",
            "           return 6",
            "       else:",
            "           if 8:",
            "               return 9",
            "           else:",
            "               if 11:",
            "                   return 12",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        actual_rule_set = ast.parse(actual_source)

        path_lines = [2, 5, 8, 11]
        one_statement_line = 11
        other_statement_line = 2

        path, one_statement = utilities.find_path_references(actual_rule_set, path_lines, one_statement_line)
        path, other_statement = utilities.find_path_references(actual_rule_set, path_lines, other_statement_line)
        mutations.swap(actual_rule_set, path, one_statement, other_statement)
        ast.fix_missing_locations(actual_rule_set)

        expected_source = "\n".join([
            "def rule_set():",
            "   if 11:",
            "       return 12",
            "   else:",
            "       if 5:",
            "           return 6",
            "       else:",
            "           if 8:",
            "               return 9",
            "           else:",
            "               if 2:",
            "                   return 3",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        expected_rule_set = ast.parse(expected_source)

        self.assertEqual(
            astor.to_source(expected_rule_set),
            astor.to_source(actual_rule_set)
        )

    def test_swap_neighbor_post_dominator(self):
        actual_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 5:",
            "           return 6",
            "       else:",
            "           if 8:",
            "               return 9",
            "           else:",
            "               if 11:",
            "                   return 12",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        actual_rule_set = ast.parse(actual_source)

        path_lines = [2, 5, 8, 11]
        one_statement_line = 5
        other_statement_line = 8

        path, one_statement = utilities.find_path_references(actual_rule_set, path_lines, one_statement_line)
        path, other_statement = utilities.find_path_references(actual_rule_set, path_lines, other_statement_line)
        mutations.swap(actual_rule_set, path, one_statement, other_statement)
        ast.fix_missing_locations(actual_rule_set)

        expected_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 8:",
            "           return 9",
            "       else:",
            "           if 5:",
            "               return 6",
            "           else:",
            "               if 11:",
            "                   return 12",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        expected_rule_set = ast.parse(expected_source)

        self.assertEqual(
            astor.to_source(expected_rule_set),
            astor.to_source(actual_rule_set)
        )

    def test_swap_non_neighbor_post_dominator(self):
        actual_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 5:",
            "           return 6",
            "       else:",
            "           if 8:",
            "               return 9",
            "           else:",
            "               if 11:",
            "                   return 12",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        actual_rule_set = ast.parse(actual_source)

        path_lines = [2, 5, 8, 11]
        one_statement_line = 5
        other_statement_line = 11

        path, one_statement = utilities.find_path_references(actual_rule_set, path_lines, one_statement_line)
        path, other_statement = utilities.find_path_references(actual_rule_set, path_lines, other_statement_line)
        mutations.swap(actual_rule_set, path, one_statement, other_statement)
        ast.fix_missing_locations(actual_rule_set)

        expected_source = "\n".join([
            "def rule_set():",
            "   if 2:",
            "       return 3",
            "   else:",
            "       if 11:",
            "           return 12",
            "       else:",
            "           if 8:",
            "               return 9",
            "           else:",
            "               if 5:",
            "                   return 6",
            "               else:",
            "                   if 14:",
            "                       return 15",
            "                   else:",
            "                       return 17"
        ])
        expected_rule_set = ast.parse(expected_source)

        self.assertEqual(
            astor.to_source(expected_rule_set),
            astor.to_source(actual_rule_set)
        )

if __name__ == '__main__':
    unittest.main()
