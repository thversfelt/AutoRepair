import pickle
import statistics
import numpy as np
import matplotlib.pyplot as plt

results_files_and_labels = {
    "prioritized_test_suite_cutoff_25_number_of_faults_3.pkl": ("Prioritized (cutoff: 25)", "red"),
    "prioritized_test_suite_cutoff_50_number_of_faults_3.pkl": ("Prioritized (cutoff: 50)", "orange"),
    "prioritized_test_suite_cutoff_75_number_of_faults_3.pkl": ("Prioritized (cutoff: 75)", "green"),
}

budget = 10
repetitions = 20

for results_file, (label, color) in results_files_and_labels.items():
    with open(results_file, "rb") as file:
        results = pickle.load(file)
    
    number_of_failing_tests_per_iteration = {}
    for iteration in range(budget):
        number_of_failing_tests_per_iteration[iteration] = []
        for repetition in results:
            archives = results[repetition]
            if iteration in archives:
                archive = archives[iteration]
                min_number_of_failing_tests = np.inf
                for individual in archive:
                    evaluation_results = archive[individual]["evaluation_results"]
                    validation_results = archive[individual]["validation_results"]
                    number_of_failing_tests = 0
                    for test_id in evaluation_results:
                        metrics_scores = evaluation_results[test_id]["metrics_scores"]
                        if  np.any(metrics_scores < 0):
                            number_of_failing_tests += 1
                    if validation_results is not None:
                        for test_id in validation_results:
                            metrics_scores = validation_results[test_id]["metrics_scores"]
                            if  np.any(metrics_scores < 0):
                                number_of_failing_tests += 1
                    if number_of_failing_tests < min_number_of_failing_tests:
                        min_number_of_failing_tests = number_of_failing_tests
                number_of_failing_tests_per_iteration[iteration].append(min_number_of_failing_tests)
            else:
                previous_min_number_of_failing_tests = number_of_failing_tests_per_iteration[iteration - 1][repetition]
                number_of_failing_tests_per_iteration[iteration].append(previous_min_number_of_failing_tests)

    # Extract the keys and values from the data dict
    x = number_of_failing_tests_per_iteration.keys()
    y = number_of_failing_tests_per_iteration.values()

    # Calculate the average and standard error for each column
    y_avg = np.array([sum(vals)/len(vals) for vals in y])
    y_err = np.array([statistics.stdev(vals)/np.sqrt(len(vals)) for vals in y])

    # Plot the data as a line with error bars
    plt.plot(x, y_avg, color=color, label=label)
    plt.fill_between(x, y_avg-y_err, y_avg+y_err, alpha=0.15, edgecolor=color, facecolor=color)

plt.grid(linestyle='--')
plt.xlim(0, budget-1)
plt.xlabel("Iteration")
plt.ylabel("Number of failing tests")
plt.title(f"Number of failing tests per iteration\n(number of faults: 3, budget: {budget}, repetitions: {repetitions})")
plt.legend()
plt.show()
print("")
                        
