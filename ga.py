# ga.py

import random

OPTIMIZE_DIR = 'optimize'

class GeneticHyperparamOptimizer:
    def __init__(self):
        # Define the hyperparameter search space
        self.hyperparam_space = {
            'learning_rate': [1e-4, 1e-3, 1e-5],
            'batch_size': [64, 128, 256, 512],
            'gamma': [0.8, 0.925, 0.95, 0.975, 0.999],
            'gae_lambda': [0.8, 0.9, 0.95],
            'n_steps': [1024, 2048, 4096], 
            'ent_coef': [0.0, 0.001, 0.00001],
            'vf_coef': [0.25, 0.5, 1.0],
            'max_grad_norm': [1.0, 5.0, 10.0]
            
        }

    def generate_individual(self):
        """
        Create an individual with random hyperparameters.
        """
        return {k: random.choice(v) for k, v in self.hyperparam_space.items()}

    def mutate(self, individual):
        """
        Mutate an individual's hyperparameters.
        """
        mutation_key = random.choice(list(individual.keys()))
        individual[mutation_key] = random.choice(self.hyperparam_space[mutation_key])
        return individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two individuals.
        """
        child = {}
        for key in self.hyperparam_space.keys():
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child

    def evaluate(self, individual, train_function, eval_function, env_fn):
        
        env_kwargs = {"n_pursuers": 8}

        # Now pass learning_rate and batch_size as separate arguments
        train_function(
            env_fn, 
            individual['learning_rate'], 
            individual['batch_size'], 
            individual['gamma'],
            individual['gae_lambda'],
            individual['n_steps'],
            individual['ent_coef'],
            individual['vf_coef'],
            individual['max_grad_norm'],
            
            model_subdir=OPTIMIZE_DIR,            
            steps=196_608, 
            **env_kwargs
        )
        avg_reward = eval_function(env_fn, num_games=10, model_subdir=OPTIMIZE_DIR, **env_kwargs)
        return avg_reward



    def run(self, train_function, eval_function, env_fn, population_size=10, generations=5):
        """
        Run the genetic algorithm.
        """
        population = [self.generate_individual() for _ in range(population_size)]
        for generation in range(generations):
            # Evaluate all individuals
            fitness_scores = [self.evaluate(individual, train_function, eval_function, env_fn) for individual in population]

            # Selection - top 50% performers
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
            population = sorted_population[:population_size//2]

            # Crossover and mutation to create new individuals
            while len(population) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                population.append(child)

            print(f"Generation {generation + 1}, Best Score: {max(fitness_scores)}")

        return sorted_population[0]
