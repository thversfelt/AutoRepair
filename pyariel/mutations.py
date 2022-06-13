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
        number.value = random.gauss(number.value, 10**order_of_magnitude)

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

def shift(rule_set: ast.Module, statement: str, references: Dict):
    possible_other_statements = [other_statement for other_statement in references if other_statement != statement]
    other_statement = random.choice(possible_other_statements)
    swap(rule_set, statement, other_statement, references)
    return statement

def swap(rule_set: ast.Module, one_statement: str, other_statement: str, references: Dict):
    # Find the function defition reference in the rule set.
    function_definition = utilities.find_function_definition_reference(rule_set)
    
    # Find the predecessors and successors of both statements.
    one_statement_reference = references[one_statement]
    one_predecessor_reference = utilities.find_statement_predecessor(rule_set, one_statement_reference)
    one_successors_reference = one_statement_reference.orelse
    
    other_statement_reference = references[other_statement]
    other_predecessor_reference = utilities.find_statement_predecessor(rule_set, other_statement_reference)
    other_successors_reference = other_statement_reference.orelse

    if other_predecessor_reference is None:
        # The other statement has no predecessors (it is the root), so to swap one and the other, make the one 
        # statement the root.
        function_definition.body[1] = one_statement_reference
    else:
        other_predecessor_reference.orelse = [one_statement_reference]

    if one_predecessor_reference is None:
        # The one statement has no predecessors (it is the root), so to swap one and the other, make the other 
        # statement the root.
        function_definition.body[1] = other_statement_reference
    else:
        one_predecessor_reference.orelse = [other_statement_reference]

    if one_predecessor_reference == other_statement_reference:
        # The other statement is the one's predecessor.
        one_statement_reference.orelse = [other_statement_reference]
        other_statement_reference.orelse = one_successors_reference
    elif other_predecessor_reference == one_statement_reference:
        # The one statement is the other's predecessor.
        other_statement_reference.orelse = [one_statement_reference]
        one_statement_reference.orelse = other_successors_reference
    else:
        # The statements are not predecessors of each other.
        one_statement_reference.orelse = other_successors_reference
        other_statement_reference.orelse = one_successors_reference
