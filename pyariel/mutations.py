import ast
import random

from typing import Dict, List
from pyariel import utilities


def modify(rule_set: ast.Module, statement: str, references: Dict):
    modifications = [
        change_threshold_value,
        change_relational_direction,
        change_arithmetic_operation
    ]
    modification = random.choice(modifications)
    
    # Obtain the statement reference from the references dictionary.
    statement_reference = references[statement]
    
    # Obtain the statement's condition.
    statement_condition = statement_reference.test
    
    # Mutate the condition.
    modification(statement_condition)
    
    # Replace the unmodified statement with the mutated statement as string key in the references dictionary.
    mutated_statement = ast.unparse(statement_reference.test)
    references[mutated_statement] = references.pop(statement)
    
    # Return the mutated statement.
    return mutated_statement

def change_threshold_value(condition: ast.Compare):
    numbers = utilities.find_numbers_references(condition)
    
    if len(numbers) > 0:
        number = random.choice(numbers)
        order_of_magnitude = utilities.order_of_magnitude(number.value)
        number.value = random.gauss(number.value, order_of_magnitude)

def change_relational_direction(condition: ast.Compare):
    relational_operator = type(condition.ops[0])
    inverse = {
        ast.Eq:     ast.NotEq(),
        ast.NotEq:  ast.Eq(),
        ast.Lt:     ast.Gt(),
        ast.Gt:     ast.Lt(),
        ast.LtE:    ast.GtE(),
        ast.GtE:    ast.LtE(),
    }
    new_relational_operator = inverse[relational_operator]
    condition.ops = [new_relational_operator]

def change_arithmetic_operation(condition: ast.Compare):
    inverse = {
        ast.Add:    ast.Sub(),
        ast.Sub:    ast.Add(),
        ast.Mult:   ast.Div(),
        ast.Div:    ast.Mult()
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
    possible_other_statements = [other_statement for other_statement in path if other_statement is not statement]
    other_statement = random.choice(possible_other_statements)
    swap(rule_set, path, statement, other_statement)

def swap(rule_set: ast.Module, path: List[ast.If], one_statement: ast.If, other_statement: ast.If):
    # Find the function defition reference in the rule set.
    function_definition = utilities.find_function_definition_reference(rule_set)
    
    # Find the predecessors and successors of the one statement.
    one_predecessor_index = path.index(one_statement) - 1
    one_predecessor = None if one_predecessor_index == -1 else path[one_predecessor_index]
    one_successors = one_statement.orelse
    
    # Find the predecessors and successors of the other statement.
    other_predecessor_index = path.index(other_statement) - 1
    other_predecessor = None if other_predecessor_index == -1 else path[other_predecessor_index]
    other_successors = other_statement.orelse

    if other_predecessor is None:
        # The other statement has no predecessors, so to swap one and the other, make the one statement the root.
        function_definition.body[1] = one_statement
    else:
        other_predecessor.orelse = [one_statement]

    if one_predecessor is None:
        # The one statement has no predecessors, so to swap one and the other, make the other statement the root.
        function_definition.body[1] = other_statement
    else:
        one_predecessor.orelse = [other_statement]

    if one_predecessor == other_statement:
        # The other statement is the one's predecessor.
        one_statement.orelse = [other_statement]
        other_statement.orelse = one_successors
    elif other_predecessor == one_statement:
        # The one statement is the other's predecessor.
        other_statement.orelse = [one_statement]
        one_statement.orelse = other_successors
    else:
        # The statements are not predecessors of each other.
        one_statement.orelse = other_successors
        other_statement.orelse = one_successors
