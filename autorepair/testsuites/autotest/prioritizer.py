import numpy as np
import pickle

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ox import random_sequence
from pymoo.operators.mutation.inversion import inversion_mutation
from pymoo.visualization.scatter import Scatter


class SceneOrdering(ElementwiseProblem):

    def __init__(self, scenes_features: np.ndarray):
        self.scenes_features = scenes_features
        self.scene_ids = list(scenes_features.keys())
        
        number_of_scenes = len(scenes_features)
        lowest_scene_index = 0
        highest_scene_index = number_of_scenes - 1
        super().__init__(n_var=number_of_scenes, n_obj=2, n_constr=0, xl=lowest_scene_index, xu=highest_scene_index)

    def _evaluate(self, x, out, *args, **kwargs):
        # Calculate the distance between each scene and its predecessor in the ordering.
        f1 = np.sum([self.distance(i, i - 1) for i in x[1:]])
        
        # Calculate the cost of each scene in the ordering.
        f2 = np.sum([self.cost(i) for i in x])
        
        # Return the two objectives values.
        out["F"] = [f1, f2]
    
    def distance(self, i, j):
        # Calculate the distance between two scenes.
        i_scene_id = self.scene_ids[i]
        j_scene_id = self.scene_ids[j]
        
        i_features = np.array(list(self.scenes_features[i_scene_id].values()))
        j_features = np.array(list(self.scenes_features[j_scene_id].values()))

        distance = np.linalg.norm(i_features - j_features) / (i + 1)
        
        # Multiply the distance by -1 to make it a minimization problem (required for pymoo).
        return -distance
    
    def cost(self, i):
        scene_id = self.scene_ids[i]
        cost = self.scenes_features[scene_id]["Number of Agents"] / (i + 1)
        return cost

class PartiallyMappedCrossover(Crossover):
    # In the crossover, an offspring 𝑜 is formed from two selected parents 𝑝1 and 𝑝2 , with the size of N, as follows: 
    # (i) select a random position 𝑐 in 𝑝1 as the cut point; (ii) the first 𝑐 elements of 𝑝1 are selected as the first 
    # 𝑐 elements of 𝑜; (iii) extract the 𝑁 − 𝑐 elements in 𝑝2 that are not in 𝑜 yet and put them as the last 𝑁 − 𝑐 
    # elements of 𝑜.
    def __init__(self, **kwargs):
        super().__init__(n_parents=2, n_offsprings=1, **kwargs)

    def _do(self, _, X, **kwargs):

        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # the array where the offsprings will be stored to
        Y = np.full(shape=(self.n_offsprings, n_matings, n_var), fill_value=-1, dtype=X.dtype)

        for i in range(n_matings):
            parent_1 = X[0, i]
            parent_2 = X[1, i]
            offspring = []
            
            # Take the first c elements of the first parent and put them in the offspring.
            cut_point = np.random.randint(0, n_var)
            for j in range(cut_point):
                offspring.append(parent_1[j])
            
            # Take the elements of the second parent that are not in the offspring and put them in the offspring, 
            # keeping them in the same order.
            for k in range(n_var):
                if parent_2[k] not in offspring:
                    offspring.append(parent_2[k])

            # Put the offspring in the offspring array.
            Y[0, i, :] = offspring[:]
            
        return Y

class MultipleMutation(Mutation):
    
    def _do(self, problem, X, **kwargs):
        mutation_operators = [self.swap_mutation, self.invert_mutation, self.insert_mutation]
        
        Y = X.copy()
        for i, y in enumerate(X):
            # Has a 0.33 chance of doing a random mutation of SWAP, INVERT or INSERT mutation.
            mutation_operator = np.random.choice(mutation_operators)
            Y[i] = mutation_operator(y)
        return Y
    
    def swap_mutation(self, y):
        # This mutation operator randomly selects two positions in a chromosome 𝑝 and swaps the index of two genes (test 
        # case indexes in the order) to generate a new offspring
        n_var = len(y)
        i = np.random.randint(0, n_var)
        j = np.random.randint(0, n_var)
        temp = y[i]
        y[i] = y[j]
        y[j] = temp
        return y
    
    def invert_mutation(self, y):
        # This mutation randomly selects a subsequence of length 𝑁 in the chromosome 𝑝 and reverses the order of the 
        # genes in the subsequence to generate a new offspring
        n_var = len(y)
        seq = random_sequence(n_var)
        return inversion_mutation(y, seq, inplace=True)
    
    def insert_mutation(self, y):
        # This mutation randomly selects a gene in the chromosome 𝑝 and moves it to another index in the solution to 
        # generate a new offspring
        n_var = len(y)
        i = np.random.randint(0, n_var)
        j = np.random.randint(0, n_var)
        gene = y[i]
        y = np.delete(y, i)
        return np.insert(y, j, gene)

def prioritize_scenes(scenes_features, population_size=100, number_of_generations=10):
    # Create a problem instance.
    problem = SceneOrdering(scenes_features)
    
    # Create an algorithm instance.
    algorithm = NSGA2(pop_size=population_size, sampling=PermutationRandomSampling(), crossover=PartiallyMappedCrossover(), mutation=MultipleMutation())
    
    # Run the algorithm.
    res = minimize(problem, algorithm, ('n_gen', number_of_generations), verbose=True)
    
    # Get the solutions and their corresponding fitness values.
    X, F = res.opt.get("X", "F")
    
    # plot the pareto front
    # Scatter(legend=True).add(F, label="Pareto-front").show()
    
    # Get the best solution (a prioritized list of indices of the scenes in the scenes features dictionary).
    best_solution = X[np.argmin(F[:, 0])]
    
    # Transform the solution into a list of prioritized scene ids.
    scene_ids = list(scenes_features.keys())
    prioritized_scene_ids = [scene_ids[i] for i in best_solution]

    return prioritized_scene_ids
