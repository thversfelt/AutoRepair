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
    threshold = condition.comparators[0]
    order_of_magnitude = utilities.order_of_magnitude(threshold.value)
    threshold.value = random.gauss(threshold.value, order_of_magnitude)


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


def shift(rule_set: ast.Module, path: List[ast.If], statement: ast.If):
    possible_targets = [node for node in path if node is not statement]
    target = random.choice(possible_targets)
    swap(rule_set, path, statement, target)


def swap(tree: ast.Module, path: List[ast.If], one: ast.If, other: ast.If):
    other_dominates = path.index(other) < path.index(one)
    if other_dominates:
        other_predecessor_index = path.index(other) - 1
        other_predecessor = None if other_predecessor_index == -1 else path[other_predecessor_index]
        other_successor = other.orelse

        one_predecessor_index = path.index(one) - 1
        one_predecessor = None if one_predecessor_index == -1 else path[one_predecessor_index]
        one_successor = one.orelse

        if other_predecessor is None:
            tree.body = [one]
        else:
            other_predecessor.orelse = [one]

        if one_predecessor is None:
            tree.body = [other]
        else:
            one_predecessor.orelse = [other]

        if one_predecessor == other:
            one.orelse = [other]
            other.orelse = one_successor
        else:
            one.orelse = other_successor
            other.orelse = one_successor
    else:
        pass  # TODO: case 2
