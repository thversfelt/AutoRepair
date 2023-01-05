import ast

from typing import List


class TestSuite:
    def __init__(self) -> None:
        pass
    
    def evaluate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        pass
    
    def validate(self, rule_set: ast.Module, parent_evaluation_results: dict = None) -> dict:
        pass
