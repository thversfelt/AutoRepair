import ast
import random
from typing import List
from pyariel import utilities


def modify(statement: ast.If):
    modifications = [
        change_threshold_value,
        change_relational_direction,
        change_arithmetic_operation
    ]
    modification = random.choice(modifications)
    modification(statement.test)


def change_threshold_value(condition: ast.Compare):
    constant = condition.comparators[0]
    order_of_magnitude = utilities.order_of_magnitude(constant.value)
    constant.value = random.gauss(constant.value, order_of_magnitude)


def change_relational_direction(condition: ast.Compare):
    relational_operator = type(condition.ops[0])
    inverse = {
        ast.Eq: ast.NotEq(),
        ast.NotEq: ast.Eq(),
        ast.Lt: ast.Gt(),
        ast.Gt: ast.Lt(),
        ast.LtE: ast.GtE(),
        ast.GtE: ast.LtE(),
    }
    new_relational_operator = inverse[relational_operator]
    condition.ops = [new_relational_operator]


def change_arithmetic_operation(condition: ast.Compare):
    inverse = {
        ast.Add: ast.Sub(),
        ast.Sub: ast.Add(),
        ast.Mult: ast.Div(),
        ast.Div: ast.Mult()
    }

    binary_operations = []
    for child in ast.iter_child_nodes(condition):
        if type(child) == ast.BinOp and type(child.op) in inverse.keys():
            binary_operations.append(child)

    if len(binary_operations) > 0:
        binary_operation = random.choice(binary_operations)
        arithmetic_operation = type(binary_operation.op)
        binary_operation.op = inverse[arithmetic_operation]


def shift(path: List[ast.If], statement: ast.If):
    possible_targets = [node for node in path if node is not statement]
    target = random.choice(possible_targets)
    swap(path, statement, target)


def swap(path: List[ast.If], one: ast.If, other: ast.If):
    dominates = path.index(other) < path.index(one)
    if dominates:
        other_predecessor_index = path.index(other) - 1
        target_predecessor = path[other_predecessor_index]
        statement_successor = one.orelse

        target_predecessor.orelse = [one]
        one.orelse = [other]
        other.orelse = statement_successor
    else:
        pass  # TODO: case 2
