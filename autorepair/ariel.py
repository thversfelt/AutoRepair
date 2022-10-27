import ast
import random
import numpy as np
import copy

from autorepair.test_suite import TestSuite
from autorepair.utilities import select
from autorepair.mutations import modify, shift
from typing import Dict, List


class Ariel:
    @staticmethod
    def repair(rule_set: ast.Module, test_suite: TestSuite, budget: int) -> None:
        results = test_suite.evaluate(rule_set)
        archive = Ariel.update_archive({}, rule_set, results)
        
        number_of_iterations = 0
        archives = {number_of_iterations: copy.copy(archive)}
        while not Ariel.solution_found(archive) and number_of_iterations < budget:
            parent, parent_results = Ariel.select_parent(archive)
            offspring = Ariel.generate_patch(parent, parent_results)
            print(ast.unparse(offspring))
            offspring_results = test_suite.evaluate(offspring)
            archive = Ariel.update_archive(archive, offspring, offspring_results)
            
            number_of_iterations += 1
            archives[number_of_iterations] = copy.copy(archive)
        
        return archives
    
    @staticmethod
    def update_archive(archive: Dict[ast.Module, dict], individual: ast.Module, results: dict) -> dict:
        """Removes dominated elitists from the archive and adds the new individual if it is not dominated."""
        metrics_scores = np.minimum.reduce([results["metrics_scores"][test_id] for test_id in results["metrics_scores"]])

        # Remove dominated elitists from the archive.
        dominated_elitists = []
        for elitist, elitist_results in archive.items():
            elitist_metrics_scores = np.minimum.reduce([elitist_results["metrics_scores"][test_id] for test_id in elitist_results["metrics_scores"]])
            if np.all(metrics_scores >= elitist_metrics_scores):
                dominated_elitists.append(elitist)
        for elitist in dominated_elitists:
            archive.pop(elitist)
            
        # Add the new individual if it is not dominated.
        dominating_elitists = []
        for elitist, elitist_results in archive.items():
            elitist_metrics_scores = np.minimum.reduce([elitist_results["metrics_scores"][test_id] for test_id in elitist_results["metrics_scores"]])
            if np.all(metrics_scores <= elitist_metrics_scores):
                dominating_elitists.append(elitist)
        if len(dominating_elitists) == 0:
            archive[individual] = results
            
        return archive
    
    @staticmethod
    def select_parent(archive: dict) -> tuple:
        potential_parents = list(archive.keys())
        parent = random.choice(potential_parents)
        parent_results = archive[parent]
        return parent, parent_results
    
    @staticmethod
    def generate_patch(individual: ast.Module, results: dict) -> ast.Module:
        path, statement = Ariel.fault_localization(results)
        mutated_path, mutated_statement = path, statement
        mutant = copy.deepcopy(individual)
        
        # Mutate the suspected statement a number of times until it is fixed.
        counter = 0
        p = random.uniform(0, 1)
        while p <= pow(0.5, counter) and mutated_statement == statement:
            mutated_path, mutated_statement = Ariel.apply_mutation(mutant, mutated_path, mutated_statement)
            counter += 1
            p = random.uniform(0, 1)
        
        return mutant
    
    @staticmethod
    def fault_localization(results: dict) -> tuple:
        test_ids = list(results["executed_paths"].keys())
        
        failing_test_ids = []
        failing_executed_paths = []
        
        passed = {}
        failed = {}
        weights = {}
        
        for test_id in test_ids:
            executed_path = results["executed_paths"][test_id]
            metrics_scores = results["metrics_scores"][test_id]
            violation_severity = abs(min(metrics_scores))
            weights[test_id] = violation_severity
            failing = True if violation_severity > 0 else False
            
            # If the test failed, add to the list of failing tests, and add its executed path to the list of executed
            # paths that were run by failing tests.
            if failing:
                failing_test_ids.append(test_id)
                failing_executed_paths.append(executed_path)
            
            for statement in executed_path:
                # Initialize the statement passed/failed counters.
                if statement not in failed:
                    failed[statement] = 0
                if statement not in passed:
                    passed[statement] = 0
                
                # Depending on whether the test failed or passed, increment the corresponding counter.
                if failing:
                    failed[statement] += 1
                else:
                    passed[statement] += 1

        # Calculate the total weight of all failing tests, as well as the amount of passed and failed tests.
        total_weight = sum(weights.values())
        total_failed = len(failing_test_ids)
        total_passed = len(test_ids) - total_failed
        
        # Throw an error if there are no failed tests, as this method should not be called in that case.
        if total_failed == 0:
            raise Exception("No failed tests.")
        
        # Initialize the suspiciousness dictionary.
        statements = list(passed.keys())
        suspiciousness = {statement: 0 for statement in statements}
        
        # Calculate suspiciousness for each statement.
        for statement in statements:
            for test_id in test_ids:
                # Determine if the statement was executed (covered) by the test.
                covers = 1 if statement in results["executed_paths"][test_id] else 0
                suspiciousness[statement] += weights[test_id] * covers / total_weight
            passed_ratio = passed[statement] / total_passed if total_passed != 0 else 0
            failed_ratio = failed[statement] / total_failed if total_failed != 0 else 0
            suspiciousness[statement] /= (passed_ratio + failed_ratio)
        
        # Select a statement using roulette wheel selection.
        suspected_statement = select(suspiciousness)
        
        # Select a path from a failing test that executed (covered) the suspected statement.
        suspected_paths = [path for path in failing_executed_paths if suspected_statement in path]
        suspected_path = random.choice(suspected_paths)
        
        return suspected_path, suspected_statement
    
    @staticmethod
    def apply_mutation(individual: ast.Module, path: List[str], statement: str) -> str:
        mutate = random.choice([modify, shift]) if len(path) > 1 else modify
        mutated_path, mutated_statement = mutate(individual, path, statement)
        return mutated_path, mutated_statement
    
    @staticmethod
    def solution_found(archive: dict) -> bool:
        if len(archive) > 1:
            return False
        else:
            elitist_results = list(archive.values())[0]
            elitist_metrics_scores = np.minimum.reduce([elitist_results["metrics_scores"][test_id] for test_id in elitist_results["metrics_scores"]])
            solution_found = True if np.all(elitist_metrics_scores == 0) else False
            return solution_found
            
