import ast
import random
from typing import List
from pyariel import utilities


def modify(statement: ast.Compare):
    modifications = [
        change_threshold_value,
        change_relational_direction,
        change_arithmetic_operation
    ]
    modification = random.choice(modifications)
    modification(statement)


def change_threshold_value(statement: ast.Compare):
    constant = statement.comparators[0]
    order_of_magnitude = utilities.order_of_magnitude(constant.value)
    constant.value = random.gauss(constant.value, order_of_magnitude)


def change_relational_direction(statement: ast.Compare):
    relational_operator = type(statement.ops[0])
    inverse = {
        ast.Eq: ast.NotEq(),
        ast.NotEq: ast.Eq(),
        ast.Lt: ast.Gt(),
        ast.Gt: ast.Lt(),
        ast.LtE: ast.GtE(),
        ast.GtE: ast.LtE(),
    }
    new_relational_operator = inverse[relational_operator]
    statement.ops = [new_relational_operator]


def change_arithmetic_operation(node: ast.Compare):
    inverse = {
        ast.Add: ast.Sub(),
        ast.Sub: ast.Add(),
        ast.Mult: ast.Div(),
        ast.Div: ast.Mult()
    }

    binary_operations = []
    for child in ast.iter_child_nodes(node):
        if type(child) == ast.BinOp and type(child.op) in inverse.keys():
            binary_operations.append(child)

    if len(binary_operations) > 0:
        binary_operation = random.choice(binary_operations)
        arithmetic_operation = type(binary_operation.op)
        binary_operation.op = inverse[arithmetic_operation]


def shift(path: List[ast.Compare], statement: ast.Compare):
    pass
