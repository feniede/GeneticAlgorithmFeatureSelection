# Base code: https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection
import numpy as np
import pandas as pd
from random import randint
import concurrent.futures
import functools
import multiprocessing
from joblib import Parallel, delayed
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from generate_partitions import load_partitions_from_json


class GeneticAlgorithmFeatureSelection:
    def __init__(self, size, n_feat, n_elite, n_parents, mutation_rate, n_gen, fitness_function, ff_params, 
                 n_feat_gen=None, patience = None, parallel = False, random_state = None, 
                 validation_function = None, vf_params = None, num_jobs=-1):
        """
        Initialize the GeneticAlgorithmFeatureSelection object.

        Parameters:
        - size (int): Size of the population.
        - n_feat (int): Number of features.
        - n_elite (int): Number of elite individuals to be preserved in each generation.
        - n_parents (int): Number of parents selected for crossover.
        - mutation_rate (float): Rate of mutation for the genetic algorithm.
        - n_gen (int): Number of generations.
        - fitness_function (callable): The fitness function to evaluate the individuals.
        - ff_params (dict): Dictionary of parameters to be passed to the fitness function.
        - n_feat_gen (int): Max number of features to select in each chromosome (only for generation). If None, there is no limit.
        - patience (int or None): Number of generations to wait for improvement in fitness (optional).
        - parallel (bool): Flag indicating whether to perform computations in parallel (optional).
        - random_state (int or None): Seed for random number generation (optional).
        - validation_function (callable or None): Optional function for validating solutions beyond the fitness evaluation.
        - vf_params (dict or None): Dictionary of parameters to be passed to the validation function.
        - num_jobs (int): Number of parallel jobs; -1 to use all available processors (optional).
        """
        
        self.size = size
        self.n_feat = n_feat
        self.n_elite = n_elite
        self.n_parents = n_parents
        self.mutation_rate = mutation_rate
        self.n_feat_gen = n_feat_gen
        self.n_gen = n_gen
        self.parallel = parallel
        self.patience = patience
        self.random_state = random_state 
        self.fitness_function = fitness_function
        self.ff_params = ff_params
        self.validation_function = validation_function
        self.vf_params = vf_params
        self.best_chromo = []
        self.best_score = []
        self.num_jobs = num_jobs

        np.random.seed(random_state)

        if parallel:
            num_cpus = multiprocessing.cpu_count()
            print(f"Number of CPU cores available: {num_cpus}")
            
    def _random_boolean_vector(self, length, num_true):
        # Ensure num_true doesn't exceed the length
        if num_true > length:
            raise ValueError("num_true cannot be greater than the length of the vector")
        
        # Create a vector of `False` with `num_true` `True` values
        vector = np.array([True] * num_true + [False] * (length - num_true))
        
        # Shuffle the array to randomize the positions of `True`
        np.random.shuffle(vector)
        
        return vector

    def initialization_of_population(self):
        if self.n_feat_gen is None:
            population = list(np.random.randint(2, size=(self.size, self.n_feat), dtype=bool))
        else:
            population = [self._random_boolean_vector(self.n_feat, self.n_feat_gen) for _ in range(self.size)]
        return population

    def fitness_score(self, population):
        scores = []
        # Selection of non-repeated chromosomes
        unique_chromosomes, indices = np.unique(np.array(population), axis=0, return_inverse=True)
        unique_chromosomes = list(unique_chromosomes)

        if self.parallel:
        
            with concurrent.futures.ThreadPoolExecutor() as executor:
                calculate_fitness_partial = functools.partial(
                    self.fitness_function,
                    **self.ff_params
                )
                scores = Parallel(n_jobs=self.num_jobs)(delayed(calculate_fitness_partial)(chromosome) for chromosome in unique_chromosomes)

        else:
            for chromosome in unique_chromosomes:
                scores.append(self.fitness_function(chromosome, **self.ff_params))

        scores, population = np.array(scores), np.array(population)
        scores = scores[indices]

        inds = np.argsort(scores)
        return list(scores[inds]), list(population[inds])

    def selection(self, pop_after_fit):
        return pop_after_fit[0:self.n_parents].copy()

    def elite(self, pop_after_fit):
        return pop_after_fit[0:self.n_elite].copy()

    def crossover(self, pop_after_sel):
        crossover_children = []
        for i in range(0, len(pop_after_sel)-1, 2):
            new_par = []
            child_1, child_2 = pop_after_sel[i], pop_after_sel[i + 1]
            new_par = np.concatenate((child_1[:len(child_1) // 2], child_2[len(child_1) // 2:]))
            crossover_children.append(new_par)
        return crossover_children

    def mutation(self, pop_after_sel):
        mutation_range = int(self.mutation_rate * self.n_feat)
        mutation_children = []
        for n in range(0, len(pop_after_sel)):
            chromo = pop_after_sel[n].copy()
            rand_posi = []
            for i in range(0, mutation_range):
                pos = randint(0, self.n_feat - 1)
                rand_posi.append(pos)
            for j in rand_posi:
                chromo[j] = not chromo[j]
            mutation_children.append(chromo)
        return mutation_children

    def generations(self):
        best_chromo = self.best_chromo
        best_score = self.best_score
        population_nextgen = self.initialization_of_population()
        for i in range(self.n_gen):
            start_time = time.time()

            scores, pop_after_fit = self.fitness_score(population_nextgen)
            # Reconstruct

            print('Best score in generation', i + 1, ':', scores[0], end="")

            # Elite children
            elite_children = self.elite(pop_after_fit)

            # Parents selection
            pop_after_sel = self.selection(pop_after_fit)

            # Crossover children
            crossover_children = self.crossover(pop_after_sel)

            # Mutation children
            mutation_children = self.mutation(pop_after_sel)

            # Generation of next generation
            population_nextgen = elite_children + crossover_children + mutation_children

            # Fill the rest of children in the population with random ones
            if len(population_nextgen) < self.size:
                if self.n_feat_gen is None:
                    population_nextgen = population_nextgen + list(np.random.randint(2, size=(self.size - len(population_nextgen), self.n_feat), dtype=bool))
                else:
                    population_nextgen = population_nextgen + [self._random_boolean_vector(self.n_feat, self.n_feat_gen) for _ in range(self.size - len(population_nextgen))]
                    
            # Termination condition
            best_fitness = scores[0]
            if i > 0 and best_fitness >= best_score[-1]:
                self.consecutive_no_improvement += 1
            else:
                self.consecutive_no_improvement = 0

            if self.consecutive_no_improvement >= self.patience:
                print("")
                print(f"Terminating early as there's no improvement for {self.patience} consecutive generations.")
                break

            best_chromo.append(pop_after_fit[0])
            best_score.append(best_fitness)

            if self.validation_function is not None:
                val_score = self.validation_function(best_chromo[-1], **self.vf_params)
                print(' -- val_score = ', val_score, end="")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f" -- {elapsed_time:.1f} seconds.", end="")

            print("")

            self.best_chromo = best_chromo
            self.best_score = best_score

        if self.validation_function is not None:
            return best_chromo, best_score, val_score
        else:
            return best_chromo, best_score
        
        
def fitness_function(chromosome, X_trains_1, X_vals, Y_trains_1, Y_vals):
    scores=[]
    for X_train_1, X_val, Y_train_1, Y_val in zip(X_trains_1, X_vals, Y_trains_1, Y_vals):
        clf = LDA()
        clf.fit(X_train_1[:, chromosome], Y_train_1)
        predictions = clf.predict_proba(X_val[:, chromosome])[:,1]
        scores.append([roc_auc_score(Y_val, predictions), average_precision_score(Y_val, predictions)])
    
    num_feat = sum(chromosome)
    total_pop = len(chromosome)
    fit_error = 1 - np.mean(scores)
    fit_val = fit_error + num_feat / total_pop
    return fit_val

def validation_function(chromosome, X_trains, X_tests, Y_trains, Y_tests):
    predictions = []
    scores = []
    for X_train, X_test, Y_train, Y_test in zip(X_trains, X_tests, Y_trains, Y_tests):
        clf = LDA()
        clf.fit(X_train[:, chromosome], Y_train)
        predictions = clf.predict_proba(X_test[:, chromosome])[:,1]
        scores.append([roc_auc_score(Y_test, predictions), average_precision_score(Y_test, predictions)])

    num_feat = sum(chromosome)
    total_pop = len(chromosome)
    fit_error = 1 - np.mean(scores)
    fit_val = fit_error + num_feat / total_pop
    return fit_val, np.mean(scores, axis=0)
        
if __name__ == '__main__':
    
    # Loading data
    database_name = "database"
    sheet_name = "database"
    X_trains, X_tests, X_trains_1, X_vals, Y_trains, Y_tests, Y_trains_1, Y_vals = load_partitions_from_json("partitions_" + sheet_name + ".json")
    
    # Parameters function
    ff_params = {'X_trains_1': X_trains_1, 'X_vals': X_vals,
                'Y_trains_1': Y_trains_1, 'Y_vals': Y_vals}

    vf_params = {'X_trains': X_trains, 'X_tests': X_tests,
                'Y_trains': Y_trains, 'Y_tests': Y_tests}
    
    # Genetic algorithm configuration
    genetic_algo = GeneticAlgorithmFeatureSelection(
        size= 200,
        n_feat=X_trains[0].shape[1],
        n_elite = 3,
        n_parents= 50, #X_trains[0].shape[1]//2,
        mutation_rate=0.05,
        n_feat_gen = 10,
        n_gen = 2000,
        fitness_function=fitness_function,
        ff_params = ff_params,
        parallel = True,
        patience = 50,
        random_state = 1234,
        validation_function = validation_function,
        vf_params = vf_params,
        num_jobs = -1)
    
    # Run the genetic algorithm
    best_chromo, best_score, val_score = genetic_algo.generations()
    
    # Get column names
    database = pd.read_excel(database_name + ".xlsx", sheet_name = sheet_name)
    selected_name_columns = database.columns[1:]
    
    # Get optimum feature subset
    best_chromosome = genetic_algo.best_chromo[-1]
    bc_tosave = pd.DataFrame(np.expand_dims(best_chromosome,axis=-1).T, columns=selected_name_columns)

    # Save the DataFrame to the existing Excel file
    with pd.ExcelWriter(database_name + ".xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        bc_tosave.to_excel(writer, sheet_name='fs_' + sheet_name, startrow=0, index=False)
