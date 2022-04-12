import ast
import copy
import random
import astor
import numpy as np

from typing import List, Tuple, Dict
from autotest.auto_test import AutoTest
from pyariel import utilities, mutations


class PyAriel:
    def run(self, rule_set: ast.Module, test_suite: AutoTest, scene_ids: List[int]) -> Dict[ast.Module, Dict]:
        results = self.evaluate(rule_set, test_suite, scene_ids)
        print(results)
        
        archive = self.update_archive({}, rule_set, results)
        
        while not self.solution_found(archive):
            parent = self.select_parent(archive)
            parent_results = archive[parent]
            
            mutant = self.generate_patch(parent, parent_results)
            print(ast.unparse(mutant))
            mutant_results = self.evaluate(mutant, test_suite, scene_ids)
            print(mutant_results)
            
            archive = self.update_archive(archive, mutant, mutant_results)

        return archive

    def evaluate(self, rule_set: ast.Module, test_suite: AutoTest, scene_ids: List[int]) -> Dict[ast.Module, Dict]:
        return test_suite.run(copy.deepcopy(rule_set), scene_ids)

    def select_parent(self, archive: Dict[ast.Module, Dict]) -> ast.Module:
        elitists = list(archive.keys())
        return random.choice(elitists)

    def update_archive(self, archive: Dict[ast.Module, Dict], rule_set: ast.Module, results: Dict) -> Dict[ast.Module, Dict]:
        metrics_scores = [scene_results["metrics_scores"] for scene_results in results.values()]
        min_metrics_scores = np.minimum.reduce(metrics_scores)

        dominated = False
        dominated_elitists = []
        for elitist, elitist_results in archive.items():
            elitist_metrics_scores = [scene_results["metrics_scores"] for scene_results in elitist_results.values()]
            elitist_min_metrics_scores = np.minimum.reduce(elitist_metrics_scores)
            
            if utilities.dominates(elitist_min_metrics_scores, min_metrics_scores):  # An archive solution dominates.
                dominated = True
                break
            elif utilities.dominates(min_metrics_scores, elitist_min_metrics_scores):  # Dominates an archive solution.
                dominated_elitists.append(elitist)

        dominates = True if len(dominated_elitists) > 0 else False
        if not dominated and dominates:
            archive[rule_set] = results
            for elitist in dominated_elitists:
                del archive[elitist]
        elif not dominated and not dominates:
            archive[rule_set] = results

        return archive

    def solution_found(self, archive: Dict[ast.Module, Dict]) -> bool:
        for results in archive.values():
            metrics_scores = [scene_results["metrics_scores"] for scene_results in results.values()]
            min_metrics_scores = np.minimum.reduce(metrics_scores)
            min_metric_score = min(min_metrics_scores)
            
            if min_metric_score == 0:
                return True
        return False
            

    def generate_patch(self, rule_set: ast.Module, results: Dict) -> ast.Module:
        mutant = copy.deepcopy(rule_set)
        path, statement = self.fault_localization(rule_set, results)
        counter = 0
        p = random.uniform(0, 1)
        while p <= pow(0.5, counter):
            self.apply_mutation(mutant, path, statement)
            counter += 1
            p = random.uniform(0, 1)
        return mutant

    def fault_localization(self, rule_set: ast.Module, results: Dict) -> Tuple[List[int], int]:
        
        paths = []  # Records the executed path of each test case (scene).
        violation = {}  # Records the violation severity of each test case.

        passed = {}  # Records the number of passed test cases that executed each statement.
        failed = {}  # Records the number of failed test cases that executed each statement.
        
        total_passed = 0  # Records the total number of passed test cases.
        total_failed = 0  # Records the total number of failed test cases.
        
        for scene_id, scene_results in results.items():
            
            # Violation severity of this test case is taken to be the absolute of the most severe objective violation.
            violation_severity = abs(min(scene_results["metrics_scores"]))
            violation[scene_id] = violation_severity
            
            # The test case contains an objective violation, if the violation severity is more than zero.
            violated = violation_severity > 0
            
            # Increment the total passed and failed counters.
            if not violated:
                total_passed += 1
            else:
                total_failed += 1
            
            # Add this test case's executed path to the list of all recorded paths.
            path = scene_results["executed_statements"]
            paths.append(path)
            
            for statement in path:
                # Initialize the passed (failed) counters of this statement, if it hasn't been already.
                if statement not in passed:
                    passed[statement] = 0
                    failed[statement] = 0

                # Increment the passed (failed) counters of this statement, if an objective violation is (not) found.
                if not violated:
                    passed[statement] += 1
                else:
                    failed[statement] += 1
        
        suspiciousness = {}  # Records the (tarantula) suspiciousness of a statement.
        
        for statement in list(set(passed) | set(failed)):
            
            total_violation = 0
            total_covered_violation = 0
            
            for scene_id, scene_results in results.items():
                
                total_violation += violation[scene_id]
                path = scene_results["executed_statements"]
                covered = statement in path
                
                if covered:
                    total_covered_violation += violation[scene_id]
            
            failed_ratio = failed[statement] / total_failed if total_failed > 0 else 0  # Prevent division by 0 error.
            passed_ratio = passed[statement] / total_passed if total_passed > 0 else 1  # Prevent division by 0 error.
            suspiciousness[statement] = (total_covered_violation / total_violation) / (passed_ratio + failed_ratio)
        
        statement = utilities.selection(suspiciousness)  # Select a random statement using RWS.
        path = random.choice([path for path in paths if statement in path])
        
        return path, statement

    def apply_mutation(self, rule_set: ast.Module, path_lines: List[int], statement_line: int):
        path, statement = utilities.find_references(rule_set, path_lines, statement_line)

        if len(path) == 1:
            mutate = mutations.modify
        else:
            mutate = random.choice([
                mutations.modify,
                mutations.shift
            ])

        mutate(rule_set, path, statement)
        ast.fix_missing_locations(rule_set)
