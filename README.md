# Waterworld Policy Evolution

This project uses Genetic Algorithms and Reinforcement Learning (PPO and SAC) to evolve policies for agents in the Waterworld environment from the PettingZoo library.

## Installation

To install the necessary dependencies, run the following command:

```sh
pip install -r requirements.txt
```

## Usage

To train a model, run the `main.py` script with the `process_to_run` variable set to `'train'`. This will train the model using the specified settings and save it to the `models/train` directory.

To optimize the hyperparameters of the model, set the `process_to_run` variable to `'optimize'`. This will run the genetic algorithm to find the best hyperparameters for the model.

## Files

- `main.py`: The main script to run for training or optimizing the model.
- `ga.py`: Contains the `GeneticHyperparamOptimizer` class which is used for optimizing the hyperparameters of the model.
- `basic.py`: Contains a basic policy for the agents in the Waterworld environment.
- `settings.py`: Contains the settings for the Waterworld environment.
