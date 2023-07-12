import ast
import math
import random

from typing import Dict, List


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
    
def find_statement_reference(rule_set: ast.Module, statement: str) -> ast.If:
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

def find_dominated_statements(rule_set: ast.Module, statement: ast.If) -> Dict[str, ast.If]:
    """Find the statements that are dominated by the statement, i.e. the statements that are nested within the statement.

    Consider the following example:

    if x > 0:
        if y > 0:
            z = 1
        else:
            z = -1

    The statement "y > 0:" is dominated by the statement "x > 0:", so find_dominated_statements(rule_set, "x > 0:") will return ["y > 0:"].

    Args:
        rule_set (ast.Module): The rule set.
        statement (ast.If): The statement.

    Returns:
        Dict[str, ast.If]: The statements that are dominated by the statement.
    """
    dominated_statements = {}
    reached_statement = False

    # Walk the tree in a depth-first manner.
    for node in ast.walk(rule_set):

        # Only consider if statements.
        if not isinstance(node, ast.If):
            continue

        # Once we reach the statement, set the reached_statement flag to True.
        if node is statement:
            reached_statement = True
            continue

        # If we have previously reached the given statement, then every statement we encounter after it is a dominated statement.
        if reached_statement:
            dominated_statements[ast.unparse(node.test)] = node
    
    return dominated_statements

def find_dominating_statements(rule_set: ast.Module, statement: ast.If) -> Dict[str, ast.If]:
    """Find the statements that dominate the statement, i.e. the statements that the given statement is nested within.

    Consider the following example:

    if x > 0:
        if y > 0:
            z = 1
        else:
            z = -1

    The statement "x > 0:" dominates the statement "y > 0:", so find_dominating_statements(rule_set, "y > 0:") will return ["x > 0:"].

    Args:
        rule_set (ast.Module): The rule set.
        statement (ast.If): The statement.

    Returns:
        Dict[str, ast.If]: The statements that dominate the statement.
    """
    dominating_statements = {}

    # Walk the tree in a depth-first manner.
    for node in ast.walk(rule_set):

        # Only consider if statements.
        if not isinstance(node, ast.If):
            continue

        # Once we reach the statement, stop.
        if node is statement:
            break
        
        # Since we are walking the tree in a depth-first manner, every statement we encounter before the given statement is a dominating statement.
        dominating_statements[ast.unparse(node.test)] = node

    return dominating_statements

def order_of_magnitude(value: float) -> float:
    """Return the order of magnitude of the value.

    Args:
        value (float): The value.

    Returns:
        float: The order of magnitude of the value.
    """
    if value == 0:
        return 1
    else:
        return 10 ** int(math.log10(abs(value)))