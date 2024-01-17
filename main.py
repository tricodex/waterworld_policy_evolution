from __future__ import annotations

import glob
import os
import time
import matplotlib.pyplot as plt

import supersuit as ss


# import SAC from stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

# import PPO from stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from pettingzoo.sisl import waterworld_v4

from ga import GeneticHyperparamOptimizer

from settings import wwargs

mdl = "PPO" # "PPO", "SAC" 

MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'

def train_waterworld(env_fn, model_name: str, model_subdir: str, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    """
    Train a PPO model using the Waterworld environment.

    Parameters:
        env_fn (function): A function that returns the Waterworld environment.
        learning_rate (float): The learning rate for the PPO algorithm.
        batch_size (int): The number of samples per update batch.
        gamma (float): The discount factor for the reward.
        gae_lambda (float): The lambda parameter for Generalized Advantage Estimation.
        n_steps (int): The number of steps to collect samples for each update.
        ent_coef (float): The coefficient for the entropy loss term.
        vf_coef (float): The coefficient for the value function loss term.
        max_grad_norm (float): The maximum gradient norm for gradient clipping.
        model_subdir (str): The subdirectory to save the trained model.
        steps (int, optional): The total number of training steps. Defaults to 10,000.
        seed (int | None, optional): The random seed for reproducibility. Defaults to 0.
        **env_kwargs: Additional keyword arguments to be passed to the environment.

    Returns:
        None
    """
    # Train a single model to play as each agent in a cooperative Parallel environment

    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")


    if model_name == "PPO":
        model = PPO(MlpPolicy, env, verbose=3, learning_rate=1e-3, batch_size=256)
    elif model_name == "SAC":
        model = SAC(MlpPolicy, env, verbose=3, learning_rate=1e-3, batch_size=256)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.learn(total_timesteps=steps)
    model_dir_path = os.path.join(MODEL_DIR, model_subdir)
    model_path = os.path.join(model_dir_path, f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    os.makedirs(model_dir_path, exist_ok=True)  # Ensure the directory and subdirectory exist
    model.save(model_path)

    print(f"Model saved to {model_path}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, model_name, model_subdir: str = TRAIN_DIR, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(os.path.join(MODEL_DIR, model_subdir, f"{env.metadata['name']}*.zip")), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    if model_name == "PPO":
        model = PPO.load(latest_policy)
    elif model_name == "SAC":
        model = SAC.load(latest_policy)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    
    if num_games == 10:
        # Plotting the total rewards
        plt.bar(rewards.keys(), rewards.values())
        plt.xlabel('Agents')
        plt.ylabel('Total Rewards')
        plt.title('Total Rewards per Agent in Waterworld Simulation')
        plt.show()

    
    return avg_reward

# Train a model
def train(env_kwargs):
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=196_608, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(env_fn, mdl, num_games=10, render_mode=None, **env_kwargs)

    # Watch a game
    eval(env_fn, mdl, num_games=1, render_mode="human", **env_kwargs)


if __name__ == "__main__":
    env_fn = waterworld_v4
    env_kwargs = wwargs
    
    process_to_run = 'train' 

    if process_to_run == 'train':
        train(env_kwargs)
        
    elif process_to_run == 'optimize':
        optimizer = GeneticHyperparamOptimizer()
        best_hyperparams = optimizer.run(
            train_waterworld, 
            eval, 
            env_fn, 
            population_size=10, #10
            generations=5 #5
        )
        print("Best Hyperparameters:", best_hyperparams)
        

    
    