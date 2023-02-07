import pickle
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

from autorepair.ariel import Ariel


abstractions = ["failing", "prioritized", "random"]
cutoffs = [47, 56, 65]
number_of_faults = 2
budget = 600

for cutoff in cutoffs:
    for abstraction in abstractions:
        # Only visualize the failing test suite for the first cutoff, as it is the same for all cutoffs.
        if abstraction == "failing" and cutoff != cutoffs[0]:
            continue
        
        filename = f"{abstraction}_test_suite_cutoff_{cutoff}_number_of_faults_{number_of_faults}.pkl"
        with open(filename, "rb") as file:
            repetitions = pickle.load(file)
        
        number_of_failing_tests_per_iteration = {}
        execution_times_per_iteration = {}
        
        for repetition_idx, checkpoints in repetitions.items():
            for iteration_idx, (execution_time, archive) in enumerate(checkpoints.items()):
                min_number_of_failing_tests = np.inf
                for results in archive.values():
                    number_of_failing_evaluation_tests = len(Ariel.failing_tests_ids(results["evaluation_results"]))
                    number_of_failing_validation_tests = len(Ariel.failing_tests_ids(results["validation_results"]))
                    number_of_failing_tests = number_of_failing_evaluation_tests + number_of_failing_validation_tests
                    min_number_of_failing_tests = min(min_number_of_failing_tests, number_of_failing_tests)
                number_of_failing_tests_per_iteration.setdefault(iteration_idx, []).append(min_number_of_failing_tests)
                execution_times_per_iteration.setdefault(iteration_idx, []).append(execution_time)
        
        # Compute the average of the execution times per iteration
        execution_times_avg = np.array([sum(vals)/len(vals) for vals in execution_times_per_iteration.values()])

        # Compute the average of the number of failing tests per iteration
        number_of_failing_tests_avg = np.array([sum(vals)/len(vals) for vals in number_of_failing_tests_per_iteration.values()])

        # Choose a random color for the plot
        color = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()))
        plt.plot(execution_times_avg, number_of_failing_tests_avg, color=color, label=f"{abstraction} (cutoff={cutoff})")
        plt.boxplot(
            list(number_of_failing_tests_per_iteration.values()), 
            positions=execution_times_avg, widths=0.05*budget, showfliers=False, patch_artist=True, 
            boxprops=dict(facecolor=color, color=color), medianprops=dict(color=color), 
            whiskerprops=dict(color=color), capprops=dict(color=color)
        )

    plt.grid(linestyle='--')

    plt.xlabel("Execution time (s)")
    plt.ylabel("Number of failing tests")
    plt.title(f"Number of failing tests vs execution time for different abstractions")
    plt.legend()
    plt.show()
    print("")
                        
