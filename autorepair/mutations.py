import ast
import random

from typing import List, Tuple
from autorepair.utilities import find_dominated_statements, find_dominating_statements, find_statement_reference, order_of_magnitude


INVERSE_RELATIONAL_OPERATOR = {
        ast.Eq:     ast.NotEq,
        ast.NotEq:  ast.Eq,
        ast.Lt:     ast.Gt,
        ast.Gt:     ast.Lt,
        ast.LtE:    ast.GtE,
        ast.GtE:    ast.LtE
}

def modify(individual: ast.Module, path: List[str], statement: str) -> Tuple[List[str], str]:
    # Find the applicable reference to the statement.
    statement_reference = find_statement_reference(individual, statement)
    threshold_values_references = [node for node in ast.walk(statement_reference.test) if isinstance(node, ast.Num)]
    threshold_value_reference = None if len(threshold_values_references) == 0 else threshold_values_references[0]
    
    # Choose between mutating the relational operator or the threshold value, or mutate the relational operator if
    # there is no threshold value.
    if random.choice([True, False]) or threshold_value_reference is None:
        # Mutate the relational operator.
        modify_relational_operator(statement_reference.test)
    else:
        # Mutate the threshold value
        modify_threshold_value(threshold_value_reference)

    # Fix the mutated individual's missing locations, which can be caused by the mutation.
    ast.fix_missing_locations(individual)

    # Determine the string representation of the mutated statement, and replace the old statement with the new one 
    # in the path.
    mutated_statement = ast.unparse(statement_reference.test)
    mutated_path = [mutated_statement if statement == path_statement else path_statement for path_statement in path]

    return mutated_path, mutated_statement
    
def modify_relational_operator(condition: ast.Compare) -> None:
    # Inverts the condition's relational operator.
    relational_operator = type(condition.ops[0])
    condition.ops = [INVERSE_RELATIONAL_OPERATOR[relational_operator]()]
    
def modify_threshold_value(threshold: ast.Num) -> None:
    # The theshold value is mutated using a gaussian distribution with a mean of at the current value and a standard
    # deviation of the order of magnitude of the current value.
    threshold.value = random.gauss(threshold.value, order_of_magnitude(threshold.value))

def shift(individual: ast.Module, path: List[str], statement: str) -> Tuple[List[str], str]:
    statement_reference = find_statement_reference(individual, statement)

    # Find the statements that are dominated by or dominate the given statement.
    dominating_statements = find_dominating_statements(individual, statement_reference)
    dominated_statements = find_dominated_statements(individual, statement_reference)
    
    # Combine the dominating and dominated statements dictionaries.
    dominating_and_dominated_statements = {**dominating_statements, **dominated_statements}

    # Choose a statement to swap with the given statement.
    target_statement = random.choice(list(dominating_and_dominated_statements.items()))
    target_statement_test = target_statement[0]
    target_statement_reference = target_statement[1]

    target_statement_test = target_statement_reference.test
    target_statement_body = target_statement_reference.body
    
    statement_test = statement_reference.test
    statement_body = statement_reference.body
    
    target_statement_reference.test = statement_test
    target_statement_reference.body = statement_body
    
    statement_reference.test = target_statement_test
    statement_reference.body = target_statement_body
    
    # Fix the mutated individual's missing locations, which can be caused by the mutation.
    ast.fix_missing_locations(individual)

    return path, statement
    
