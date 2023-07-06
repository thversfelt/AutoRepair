import ast
from typing import Dict
import numpy as np

from autorepair.ariel import Ariel
from ast import List

from autorepair.testsuites.test_suites import TestSuite


class AutoRepair(Ariel):
     @staticmethod
     def update_archive(archive: Dict[ast.Module, dict], rule_set: ast.Module, test_suite: TestSuite, evaluation_results: dict) -> dict:
          """Removes dominated elitists from the archive and adds the new individual if it is not dominated."""
          metrics_scores = np.minimum.reduce([evaluation_results[test_id]["metrics_scores"] for test_id in evaluation_results])

          # Sum the metrics scores to obtain a total metric score, which will be used as an additional objective. In Ariel, when two individuals have the same metrics scores, one might have many test cases which fail, while the other might have few test cases which fail. The individual with fewer test cases which fail is preferred, as it is more likely to be a correct solution.
          total_metrics_score = np.sum(metrics_scores)

          # Add the total metrics score to the metrics scores.
          metrics_scores = np.append(metrics_scores, total_metrics_score)

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