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
import datetime

mdl = "PPO"  # Choose "PPO" or "SAC"

MODEL_DIR = 'models'
TRAIN_DIR = 'train'
OPTIMIZE_DIR = 'optimize'

def train_waterworld(env_fn, model_name, model_subdir, steps=50_000, seed=None, **hyperparam_kwargs):
    if 'n_steps' in hyperparam_kwargs:
        hyperparam_kwargs['n_steps'] = int(hyperparam_kwargs['n_steps'])
    if 'batch_size' in hyperparam_kwargs:
        hyperparam_kwargs['batch_size'] = int(hyperparam_kwargs['batch_size'])
    if 'buffer_size' in hyperparam_kwargs:
        hyperparam_kwargs['buffer_size'] = int(hyperparam_kwargs['buffer_size'])
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
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(
            glob.glob(os.path.join(MODEL_DIR, model_subdir, f"{env.metadata['name']}*.zip")), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    reward_counters = {'poison_reward': 0, 'food_reward': 0, 'encounter_reward': 0, 'thrust_penalty': 0}
    rewards = {agent: 0 for agent in env.possible_agents}

    if model_name == "PPO":
        model = PPO.load(latest_policy)
        for i in range(num_games):
            env.reset(seed=i)
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                for a in env.agents:
                    rewards[a] += env.rewards[a]
                # Update specific reward counters
                update_reward_counters(info, reward_counters)
                if termination or truncation:
                    break
                else:
                    act = model.predict(obs, deterministic=True)[0]
                env.step(act)
        env.close()

    elif model_name == "SAC":
        model = SAC.load(latest_policy)
        for i in range(num_games):
            env.reset(seed=i)
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                for a in env.agents:
                    rewards[a] += env.rewards[a]
                # Update specific reward counters
                update_reward_counters(info, reward_counters)
                if termination or truncation:
                    action = None
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    action = action.reshape(env.action_space(agent).shape) if model_name == "SAC" else action
                env.step(action)
        env.close()
    
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")

    # Plotting total rewards
    os.makedirs('plots/eval', exist_ok=True)
    if num_games == 10:
        plt.figure()
        plt.bar(rewards.keys(), rewards.values())
        plt.xlabel('Agents')
        plt.ylabel('Total Rewards')
        plt.title('Total Rewards per Agent in Waterworld Simulation')
        plot_name = f'{mdl}_{process_to_run}_rewards_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
        plt.savefig(f'plots/eval/{plot_name}')

    # Plotting specific rewards and penalties
    if num_games == 10:
        plt.figure()
        plt.bar(reward_counters.keys(), reward_counters.values())
        plt.xlabel('Type')
        plt.ylabel('Total Value')
        plt.title('Distribution of Different Rewards and Penalties in Waterworld Simulation')
        distributed_plot_name = f'distributed_{mdl}_{process_to_run}_rewards_plot_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
        plt.savefig(f'plots/eval/{distributed_plot_name}')

    return avg_reward

def update_reward_counters(info, reward_counters):
    if 'poison_reward' in info:
        reward_counters['poison_reward'] += info['poison_reward']
    if 'food_reward' in info:
        reward_counters['food_reward'] += info['food_reward']
    if 'encounter_reward' in info:
        reward_counters['encounter_reward'] += info['encounter_reward']
    if 'thrust_penalty' in info:
        reward_counters['thrust_penalty'] += info['thrust_penalty']


# Train a model
def run_train():
    # still arbitrary episodes and episode lengths
    episodes, episode_lengths = 200, 1000
    total = episode_lengths*episodes

    # Train the waterworld environment with the specified model and settings
    train_waterworld(env_fn, mdl, TRAIN_DIR, steps=total, seed=0)
    
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
            population_size=30,
            generations=20
        )
        print("Best Hyperparameters:", best_hyperparams)
        

    
    