import ast
import math
import random

from typing import Dict


def select(suspiciousness: Dict[str, float]) -> str:
    """Roullette wheel selection.

    Args:
        suspiciousness (Dict[str, float]): A dictionary of suspiciousness values for each statement.

    Returns:
        str: The selected suspected statement.
    """
    total_value = sum(suspiciousness.values())
    selection_value = random.uniform(0, total_value)
    value = 0
    for statement, suspiciousness in suspiciousness.items():
        if value + suspiciousness > selection_value:
            return statement
        value += suspiciousness
    assert False, "Shouldn't get here"
    
def find_reference(rule_set: ast.Module, statement: str) -> ast.If:
    """Find the reference to the statement in the rule set.

    Args:
        rule_set (ast.Module): The rule set.
        statement (str): The statement to find.

    Returns:
        ast.If: The reference to the statement.
    """
    for node in ast.walk(rule_set):
        if isinstance(node, ast.If) and ast.unparse(node.test) == statement:
            return node
    raise Exception("Statement not found.")

def order_of_magnitude(value: float) -> float:
    """Return the order of magnitude of the value.

    Args:
        value (float): The value.

    Returns:
        float: The order of magnitude of the value.
    """
    return 10 ** int(math.log10(value))