import ast
import math
import random

from typing import Dict, List, Tuple, Any

def selection(population: Dict[int, float]) -> int:
    """Roulette Wheel Selection"""
    total_value = sum(population.values())
    selection_value = random.uniform(0, total_value)
    current_value = 0
    for key, value in population.items():
        current_value += value
        if current_value >= selection_value:
            return key

def dominates(one: List[float], other: List[float]) -> bool:
    for i in range(len(one)):
        if other[i] > one[i]:
            return False  # One is dominated by at least one objective of the other

    for i in range(len(one)):
        if one[i] > other[i]:
            return True  # The other does not dominate one, and one dominates at least one objectives of the other.

    return False  # The other does not dominate one, and one does not dominate any objective of the other.

def order_of_magnitude(number: float) -> int:
    if number == 0:
        return 10
    else:
        return math.floor(math.log(abs(number), 10))

def find_path_references(rule_set: ast.Module, path_lines: List[int], statement_line: int) -> Tuple[List[ast.If], ast.If]:
    class ReferencesFinder(ast.NodeVisitor):
        def __init__(self):
            self.path = dict.fromkeys(path_lines, None)
            super().__init__()

        def visit_If(self, node: ast.If) -> Any:
            if node.lineno in self.path.keys():
                self.path[node.lineno] = node
            self.generic_visit(node)

    finder = ReferencesFinder()
    finder.visit(rule_set)
    path = list(finder.path.values())
    statement = finder.path[statement_line]
    return path, statement

def find_function_definition_reference(rule_set: ast.Module) -> ast.FunctionDef:
    class ReferenceFinder(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self.function_definition = node
            self.generic_visit(node)
        
    finder = ReferenceFinder()
    finder.visit(rule_set)
    return finder.function_definition
    