import unittest
import ast
import astor

from pyariel import utilities, mutations


class TestMutations(unittest.TestCase):
    def test_swap_dominator(self):
        source = """
if 2:
    return 12
else:
    if 3:
        return 11
    else:
        if 4:
            return 10
        else:
            if 5:
                return 9
            else:
                if 6:
                    return 8
                else:
                    return 7
        """
        tree = ast.parse(source)
        path = [2, 5, 8]
        statement = 8
        target = 5

        path_references, statement_reference = utilities.find_references(tree, path, statement)
        path_references, target_reference = utilities.find_references(tree, path, target)
        mutations.swap(path_references, statement_reference, target_reference)

        expected_source = """
if 2:
    return 12
else:
    if 4:
        return 10
    else:
        if 3:
            return 11
        else:
            if 5:
                return 9
            else:
                if 6:
                    return 8
                else:
                    return 7
                """
        expected_tree = ast.parse(expected_source)

        self.assertEqual(astor.to_source(tree), astor.to_source(expected_tree))  # add assertion here


if __name__ == '__main__':
    unittest.main()
