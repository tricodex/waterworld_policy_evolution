# main.py

from __future__ import annotations
import glob
import os
import time
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from pettingzoo.sisl import waterworld_v4
from ga import GeneticHyperparamOptimizer
from settings import env_kwargs

mdl = "SAC"  # Choose "PPO" or "SAC"

MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'

def train_waterworld(env_fn, model_name, model_subdir, steps=10_000, seed=None, **hyperparam_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    if model_name == "PPO":
        model = PPO(PPOMlpPolicy, env, verbose=3, **hyperparam_kwargs)
    elif model_name == "SAC":
        #policy_kwargs = {"net_arch": [dict(pi=[400, 300], qf=[400, 300])]} # policy_kwargs=policy_kwargs
        model = SAC(SACMlpPolicy, env, verbose=3, **hyperparam_kwargs)
    
    elif model_name == 'SAC' and process_to_run == "train":
        model = SAC(SACMlpPolicy, env, verbose=3, buffer_size=10000 **hyperparam_kwargs)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.learn(total_timesteps=steps)
    model_dir_path = os.path.join(MODEL_DIR, model_subdir)
    model_path = os.path.join(model_dir_path, f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    os.makedirs(model_dir_path, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()

def eval(env_fn, model_name, model_subdir=TRAIN_DIR, num_games=100, render_mode=None):
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

    model = None
    if model_name == "PPO":
        model = PPO.load(latest_policy)
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

    elif model_name == "SAC":
        model = SAC.load(latest_policy)
        rewards = {agent: 0 for agent in env.possible_agents}
        for i in range(num_games):
            env.reset(seed=i)
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    if model_name == "SAC":
                        # For SAC, ensure action is reshaped or formatted correctly
                        action = action.reshape(env.action_space(agent).shape)
                env.step(action)
                for a in env.agents:
                    rewards[a] += env.rewards[a]
        env.close()
    
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")

    if num_games == 10:
        plt.bar(rewards.keys(), rewards.values())
        plt.xlabel('Agents')
        plt.ylabel('Total Rewards')
        plt.title('Total Rewards per Agent in Waterworld Simulation')
        plt.show()

    return avg_reward


# Train a model
def run_train():
    # Train the waterworld environment with the specified model and settings
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=196_608, seed=0)
    
    # Evaluate the trained model against a random agent for 10 games without rendering
    eval(env_fn, mdl, num_games=10, render_mode=None)
    
    # Evaluate the trained model against a random agent for 1 game with rendering
    eval(env_fn, mdl, num_games=1, render_mode="human")

if __name__ == "__main__":
    env_fn = waterworld_v4  
    process_to_run = 'optimize' 

    if process_to_run == 'train':
        run_train()
    elif process_to_run == 'optimize':
        optimizer = GeneticHyperparamOptimizer(model_name=mdl)
        best_hyperparams = optimizer.run(
            train_waterworld, 
            eval, 
            env_fn, 
            population_size=4,
            generations=2
        )
        print("Best Hyperparameters:", best_hyperparams)
        

    
    