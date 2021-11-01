import unittest
import ast
import astor

from pyariel import utilities, mutations


class TestMutations(unittest.TestCase):
    def test_swap_neighbor_dominator(self):
        actual_source = "\n".join([
            "if 1:",
            "    return 2",
            "else:",
            "    if 4:",
            "        return 5",
            "    else:",
            "        if 7:",
            "            return 8",
            "        else:",
            "            if 10:",
            "                return 11",
            "            else:",
            "                if 13:",
            "                    return 14",
            "                else:",
            "                    return 16"
        ])
        actual_tree = ast.parse(actual_source)

        path_lines = [1, 4, 7]
        target_line = 4
        statement_line = 7

        path, statement = utilities.find_references(actual_tree, path_lines, statement_line)
        path, target = utilities.find_references(actual_tree, path_lines, target_line)
        mutations.swap(actual_tree, path, statement, target)
        ast.fix_missing_locations(actual_tree)

        expected_source = "\n".join([
            "if 1:",
            "    return 2",
            "else:",
            "    if 7:",
            "        return 8",
            "    else:",
            "        if 4:",
            "            return 5",
            "        else:",
            "            if 10:",
            "                return 11",
            "            else:",
            "                if 13:",
            "                    return 14",
            "                else:",
            "                    return 16"
        ])
        expected_tree = ast.parse(expected_source)

        self.assertEqual(
            astor.to_source(expected_tree),
            astor.to_source(actual_tree)
        )

    def test_swap_non_neighbor_dominator(self):
        actual_source = "\n".join([
            "if 1:",
            "    return 2",
            "else:",
            "    if 4:",
            "        return 5",
            "    else:",
            "        if 7:",
            "            return 8",
            "        else:",
            "            if 10:",
            "                return 11",
            "            else:",
            "                if 13:",
            "                    return 14",
            "                else:",
            "                    return 16"
        ])
        actual_tree = ast.parse(actual_source)

        path_lines = [1, 4, 7, 10]
        target_line = 1
        statement_line = 10

        path, statement = utilities.find_references(actual_tree, path_lines, statement_line)
        path, target = utilities.find_references(actual_tree, path_lines, target_line)
        mutations.swap(actual_tree, path, statement, target)
        ast.fix_missing_locations(actual_tree)

        expected_source = "\n".join([
            "if 10:",
            "    return 11",
            "else:",
            "    if 4:",
            "        return 5",
            "    else:",
            "        if 7:",
            "            return 8",
            "        else:",
            "            if 1:",
            "                return 2",
            "            else:",
            "                if 13:",
            "                    return 14",
            "                else:",
            "                    return 16"
        ])
        expected_tree = ast.parse(expected_source)

        self.assertEqual(
            astor.to_source(expected_tree),
            astor.to_source(actual_tree)
        )


if __name__ == '__main__':
    unittest.main()
