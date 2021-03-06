import ast
import math
import random
import numpy as np

from typing import Dict, List, Tuple, Any
from deap.tools._hypervolume import hv

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

def get_pareto_front(archive: Dict[ast.Module, Dict]) -> List[List[float]]:
    front = []

    for results in archive.values():
        metrics_scores = [scene_results["metrics_scores"] for scene_results in results.values()]
        min_metrics_scores = np.minimum.reduce(metrics_scores)
        front.append(min_metrics_scores)
    
    return front

def calculate_hypervolume(front: List[List[float]]) -> float:
    """Calculate the hypervolume of a front"""
    reference_point = len(front[0]) * [0.1]
    return hv.hypervolume(front, reference_point)

def find_statements_references(rule_set: ast.Module, path: List[str]) -> Tuple[List[ast.If], ast.If]:
    class ReferencesFinder(ast.NodeVisitor):
        def __init__(self):
            self.references = {}
            super().__init__()

        def visit_If(self, node: ast.If) -> Any:
            condition = ast.unparse(node.test)
            if condition in path:
                self.references[condition] = node
            self.generic_visit(node)

    finder = ReferencesFinder()
    finder.visit(rule_set)
    return finder.references

def find_function_definition_reference(rule_set: ast.Module) -> ast.FunctionDef:
    class ReferenceFinder(ast.NodeVisitor):
        def __init__(self):
            self.function_definition = None
            super().__init__()
        
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self.function_definition = node
            self.generic_visit(node)
        
    finder = ReferenceFinder()
    finder.visit(rule_set)
    return finder.function_definition
    
def find_arithmetic_operations_references(condition: ast.Compare) -> List[ast.BinOp]:
    class ReferenceFinder(ast.NodeVisitor):
        def __init__(self):
            self.operations = []
            super().__init__()
        
        def visit_BinOp(self, node: ast.BinOp) -> Any:
            self.operations.append(node)
            self.generic_visit(node)
        
    finder = ReferenceFinder()
    finder.visit(condition)
    return finder.operations

def find_numbers_references(condition: ast.Compare) -> List[ast.Num]:
    class ReferenceFinder(ast.NodeVisitor):
        def __init__(self):
            self.numbers = []
            super().__init__()
        
        def visit_Num(self, node: ast.Num) -> Any:
            self.numbers.append(node)
            self.generic_visit(node)
        
    finder = ReferenceFinder()
    finder.visit(condition)
    return finder.numbers

def find_statement_predecessor(rule_set: ast.Module, statement: ast.If) -> ast.If:
    class PredecessorFinder(ast.NodeVisitor):
        def __init__(self):
            self.predecessor = None
            super().__init__()

        def visit_If(self, node: ast.If) -> Any:
            if node is statement:
                return
            self.predecessor = node
            self.generic_visit(node)
            
    finder = PredecessorFinder()
    finder.visit(rule_set)
    return finder.predecessor