import ast
import copy
import inspect
import random
from typing import Callable, List, Tuple, Dict
import astor
import numpy as np
from pyariel import utilities, instrumentation, mutations


class PyAriel:
    def run(self, rule_set: Callable, test_suite: List[Callable], test_scope: Dict) -> Dict[ast.Module, List[float]]:
        source = inspect.getsource(rule_set)  # Retrieve the source code of the rule set.
        rule_set_ast = ast.parse(source)  # Parse the source of the rule set into an AST.

        archive = self.update_archive({}, rule_set_ast, test_suite, test_scope)
        while [0.0] * len(test_suite) not in archive.values():
            parent_rule_set = self.select_parent(archive)
            mutated_rule_set = self.generate_patch(parent_rule_set, test_suite, test_scope)
            archive = self.update_archive(archive, mutated_rule_set, test_suite, test_scope)

            print('----ARCHIVE----')
            for solution, solution_objectives_scores in archive.items():
                print(solution_objectives_scores)
                print(astor.to_source(solution))

        return archive

    def select_parent(self, archive: Dict[ast.Module, List[float]]) -> ast.Module:
        return random.choice(list(archive.keys()))

    def update_archive(self, archive: Dict[ast.Module, List[float]], rule_set: ast.Module, test_suite: List[Callable],
                       scope: Dict) -> Dict[ast.Module, List[float]]:
        code = compile(rule_set, '<ast>', 'exec')  # Compile the instrumented AST of the rule set.
        exec(code, scope)  # Execute the compiled AST of the rule set in the given scope.
        callable_rule_set = scope.get('rule_set')  # Extract the rule set function definition from the scope.

        min_objectives_scores = None  # Records the lowest scores for each objective.
        for test in test_suite:  # For each test case in the test suite, do:
            objectives_scores = test(callable_rule_set)  # Run the test case and record the result.
            if min_objectives_scores is None:
                min_objectives_scores = objectives_scores
            else:
                min_objectives_scores = list(np.minimum(min_objectives_scores, objectives_scores))

        dominated = False
        dominated_solutions = []
        for solution, solution_objectives_scores in archive.items():
            if utilities.dominates(solution_objectives_scores, min_objectives_scores):  # An archive solution dominates.
                dominated = True
                break
            elif utilities.dominates(min_objectives_scores,
                                     solution_objectives_scores):  # Dominates an archive solution.
                dominated_solutions.append(solution)

        dominates = True if len(dominated_solutions) > 0 else False
        if not dominated and dominates:
            archive[rule_set] = min_objectives_scores
            for solution in dominated_solutions:
                del archive[solution]
        elif not dominated and not dominates:
            archive[rule_set] = min_objectives_scores

        return archive

    def generate_patch(self, rule_set: ast.Module, test_suite: List[Callable], scope: Dict) -> ast.Module:
        mutated_rule_set = copy.deepcopy(rule_set)
        path, statement = self.fault_localization(rule_set, test_suite, scope)
        counter = 0
        p = random.uniform(0, 1)
        while p <= pow(0.5, counter):
            self.apply_mutation(mutated_rule_set, path, statement)
            counter += 1
            p = random.uniform(0, 1)
        return mutated_rule_set

    def fault_localization(self, rule_set: ast.Module, test_suite: List[Callable], test_scope: Dict) -> Tuple[
        List[int], int]:
        instrumented_rule_set = copy.deepcopy(
            rule_set)  # Replace the original rule set with a deep copy for instrumentation.
        instrumentation.Instrumenter().visit(instrumented_rule_set)  # Instrument the rule set.
        ast.fix_missing_locations(instrumented_rule_set)  # Fix the missing line numbers and other fields in the AST.

        code = compile(instrumented_rule_set, '<ast>', 'exec')  # Compile the instrumented AST of the rule set.
        exec(code, test_scope)  # Execute the compiled AST of the rule set in the given scope.
        callable_rule_set = test_scope['rule_set']  # Extract the rule set function definition from the scope.

        executed_paths = []  # Records all executed paths during execution of all test cases in the test suite.
        passed = {}  # Records the number of passed test cases that executed each statement.
        failed = {}  # Records the number of failed test cases that executed each statement.
        total_passed = 0  # Records the total number of passed test cases.
        total_failed = 0  # Records the total number of failed test cases.

        for test in test_suite:  # For each test case in the test suite, do:
            objectives_scores = test(callable_rule_set)  # Run the test case and record the result.
            executed_path = test_scope['executed_path']  # Extract the executed path variable from the scope.
            executed_paths.append(executed_path)  # Add the executed path to the list of all executed paths.

            test_passed = True if -1.0 not in objectives_scores else False
            total_passed += test_passed  # Increment the total amount of passed tests.
            total_failed += (not test_passed)  # Increment the total amount of failed tests.

            for statement in executed_path:  # For each of the executed statements of the executed path, do:
                passed_count = passed.get(statement) if statement in passed else 0
                failed_count = failed.get(statement) if statement in failed else 0
                passed[statement] = passed_count + test_passed  # Increment the amount of passed tests for it.
                failed[statement] = failed_count + (not test_passed)  # Increment the amount of failed tests for it.

        suspiciousness = {}
        for statement in list(set(passed) | set(failed)):  # Determine the suspiciousness of each executed statement.
            suspiciousness[statement] = utilities.suspiciousness(statement, passed, failed, total_passed, total_failed)

        statement = utilities.selection(suspiciousness)  # Select a random statement using RWS.
        possible_paths = [executed_path for executed_path in executed_paths if statement in executed_path]
        path = random.choice(possible_paths)  # Choose a random executed path that contains the selected statement.
        return path, statement

    def apply_mutation(self, rule_set: ast.Module, path: List[int], statement: int):
        path_references, statement_reference = utilities.find_references(rule_set, path, statement)
        mutations.modify(statement_reference)
        ast.fix_missing_locations(rule_set)
