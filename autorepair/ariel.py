import ast
import random
import time
import numpy as np
import copy

from autorepair.testsuites.test_suites import TestSuite
from autorepair.utilities import select
from autorepair.mutations import modify, shift
from typing import Dict, List


class Ariel:
    @staticmethod
    def repair(rule_set: ast.Module, test_suite: TestSuite, evaluation_tests_ids: List[int], budget: int, validate: bool=False, evaluate_failing_tests: bool=False) -> None:
        
        evaluation_start_time = time.time()
        evaluation_results = test_suite.evaluate(rule_set, evaluation_tests_ids)
        evaluation_time = time.time() - evaluation_start_time
        
        evaluation_tests_ids = Ariel.failing_tests_ids(evaluation_results) if evaluate_failing_tests else evaluation_tests_ids
        
        archive = Ariel.update_archive({}, rule_set, test_suite, evaluation_results)
        checkpoint = copy.deepcopy(archive)
        checkpoints = {evaluation_time: checkpoint}
        
        while not Ariel.solution_found(archive) and evaluation_time < budget:
            parent, parent_evaluation_results = Ariel.select_parent(archive)
            
            offspring = Ariel.generate_patch(parent, parent_evaluation_results)
            evaluation_start_time = time.time()
            offspring_evaluation_results = test_suite.evaluate(offspring, evaluation_tests_ids)
            evaluation_time += time.time() - evaluation_start_time
            
            archive = Ariel.update_archive(archive, offspring, test_suite, offspring_evaluation_results)
            checkpoint = copy.deepcopy(archive)
            checkpoints[evaluation_time] = checkpoint
        
        return checkpoints
    
    @staticmethod
    def update_archive(archive: Dict[ast.Module, dict], rule_set: ast.Module, test_suite: TestSuite, evaluation_results: dict) -> dict:
        """Removes dominated elitists from the archive and adds the new individual if it is not dominated."""
        metrics_scores = np.minimum.reduce([evaluation_results[test_id]["metrics_scores"] for test_id in evaluation_results])

        # Determine which elitists are dominated by the individual.
        dominated_elitists = []
        for elitist, elitist_results in archive.items():
            elitist_metrics_scores = np.minimum.reduce([elitist_results["evaluation_results"][test_id]["metrics_scores"] for test_id in elitist_results["evaluation_results"]])
            if np.all(metrics_scores >= elitist_metrics_scores):
                dominated_elitists.append(elitist)

        # Determine which elitists dominate the individual.
        dominating_elitists = []
        for elitist, elitist_results in archive.items():
            elitist_metrics_scores = np.minimum.reduce([elitist_results["evaluation_results"][test_id]["metrics_scores"] for test_id in elitist_results["evaluation_results"]])
            if np.all(metrics_scores <= elitist_metrics_scores):
                dominating_elitists.append(elitist)
        
        # If the individual is dominated by any elitists, return the archive without adding the individual.
        if len(dominating_elitists) > 0:
            return archive
        
        # Remove the dominated elitists from the archive.
        for elitist in dominated_elitists:
            archive.pop(elitist)
        
        evaluation_tests_ids = [test_id for test_id in evaluation_results]
        validation_results = test_suite.validate(rule_set, evaluation_tests_ids)
        
        # Add the individual to the archive.
        archive[rule_set] = {
            "evaluation_results": evaluation_results,
            "validation_results": validation_results
        }
        
        return archive
    
    @staticmethod
    def select_parent(archive: dict) -> tuple:
        potential_parents = list(archive.keys())
        parent = random.choice(potential_parents)
        parent_evaluation_results = archive[parent]["evaluation_results"]
        return parent, parent_evaluation_results
    
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
        test_ids = list(results.keys())
        
        failing_test_ids = []
        failing_executed_paths = set()
        
        passed = {}
        failed = {}
        weights = {}
        
        for test_id in test_ids:
            executed_path = results[test_id]["executed_paths"]
            metrics_scores = results[test_id]["metrics_scores"]
            violation_severity = abs(min(metrics_scores))
            weights[test_id] = violation_severity
            failing = True if violation_severity > 0 else False
            
            # If the test failed, add to the list of failing tests, and add its executed path to the list of executed
            # paths that were run by failing tests.
            if failing:
                failing_test_ids.append(test_id)
                failing_executed_paths.add(tuple(executed_path))
            
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
                covers = 1 if statement in results[test_id]["executed_paths"] else 0
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
            elitist_metrics_scores = np.minimum.reduce([elitist_results["evaluation_results"][test_id]["metrics_scores"] for test_id in elitist_results["evaluation_results"]])
            solution_found = True if np.all(elitist_metrics_scores == 0) else False
            return solution_found
    
    @staticmethod
    def failing_tests_ids(results: dict = None) -> List[int]:
        failing_tests_ids = []
        for test_id, test_results in results.items():
            metrics_scores = test_results["metrics_scores"]
            if np.any(metrics_scores < 0):
                failing_tests_ids.append(test_id)
        return failing_tests_ids
