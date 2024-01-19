# ga.py

import random
import logging
import matplotlib.pyplot as plt
from settings import hyperparam_space_ppo, hyperparam_space_sac, env_kwargs
import datetime
import os

OPTIMIZE_DIR = 'optimize'

class GeneticHyperparamOptimizer:
    def __init__(self, model_name):
        logging.basicConfig(filename='genetic_algo.log', level=logging.INFO)
        self.model_name = model_name

        # Select hyperparameter space based on model name
        if model_name == "PPO":
            self.hyperparam_space = hyperparam_space_ppo
        elif model_name == "SAC":
            self.hyperparam_space = hyperparam_space_sac
        else:
            raise ValueError("Invalid model name")

    def generate_individual(self):
        """
        Create an individual with random hyperparameters.
        """
        return {k: random.choice(v) for k, v in self.hyperparam_space.items()}

    def mutate(self, individual):
        """
        Mutate multiple hyperparameters of an individual.
        """
        num_mutations = random.randint(1, len(self.hyperparam_space))  # Number of hyperparameters to mutate
        mutation_keys = random.sample(list(individual.keys()), num_mutations)
        for mutation_key in mutation_keys:
            # Gaussian mutation example
            current_value = individual[mutation_key]
            mutation_range = (max(self.hyperparam_space[mutation_key]) - min(self.hyperparam_space[mutation_key])) * 0.1
            new_value = current_value + random.gauss(0, mutation_range)
            # Clipping new value to hyperparameter's range
            new_value = max(min(new_value, max(self.hyperparam_space[mutation_key])), min(self.hyperparam_space[mutation_key]))
            individual[mutation_key] = new_value
        return individual


    def crossover(self, parent1, parent2):
        """
        Perform crossover between two individuals with fitness consideration.
        """
        child = {}
        for key in self.hyperparam_space.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child


    def evaluate(self, individual, train_function, eval_function, env_fn):

        # Select hyperparameters based on the model
        if self.model_name == "PPO":
            # Filter out only those hyperparameters that are relevant for PPO
            ppo_params = {k: individual[k] for k in individual if k in hyperparam_space_ppo}
            hyperparams = ppo_params
        elif self.model_name == "SAC":
            # Filter out only those hyperparameters that are relevant for SAC
            sac_params = {k: individual[k] for k in individual if k in hyperparam_space_sac}
            hyperparams = sac_params
        else:
            raise ValueError("Invalid model name")

        # Pass the relevant hyperparameters to the train function
        train_function(env_fn, self.model_name, OPTIMIZE_DIR, steps=196_608, seed=0, **hyperparams)

        # Evaluate the trained model
        avg_reward = eval_function(env_fn, self.model_name, model_subdir=OPTIMIZE_DIR, num_games=10)
        logging.info(f"Evaluating Individual: {individual}, Avg Reward: {avg_reward}")
        return avg_reward




    def run(self, train_function, eval_function, env_fn, population_size=10, generations=5, elitism_size=2):
        """
        Run the genetic algorithm with elitism.
        """
        population = [self.generate_individual() for _ in range(population_size)]
        best_scores = []
        for generation in range(generations):
            fitness_scores = [self.evaluate(individual, train_function, eval_function, env_fn) for individual in population]
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

            # Elitism - carry over some best individuals
            next_generation = sorted_population[:elitism_size]
            
            # Crossover and mutation for the rest
            while len(next_generation) < population_size:
                parent1, parent2 = random.sample(sorted_population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation
            best_score = max(fitness_scores)
            best_scores.append(best_score)
            logging.info(f"Generation {generation + 1}, Best Score: {best_score}")
        
        self.plot_performance(best_scores)
        return sorted_population[0]

    def plot_performance(self, best_scores):
        plt.plot(best_scores)
        plt.xlabel('Generation')
        plt.ylabel('Best Score')
        plt.title('Best Score Evolution')

        # Directory where plots will be saved
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Formatting the filename with the current date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plot_filename = os.path.join(plots_dir, f'performance_plot_{current_time}.png')

        plt.savefig(plot_filename)
        # plt.show()  # Removed to prevent halting the process
