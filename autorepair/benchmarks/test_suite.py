import ast

from typing import List


class TestSuite:
    def __init__(self) -> None:
        pass
    
    def evaluate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        pass
    
    def validate(self, rule_set: ast.Module, evaluation_tests_ids: List[int]) -> dict:
        pass
    
    def run(self, rule_set: ast.Module, tests_ids: List[int]) -> dict:
        pass

