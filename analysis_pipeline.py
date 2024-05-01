import os
import datetime
import pickle
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import waterworld_v4 
import seaborn as sns
from settings import env_kwargs
from analysis import Analysis
from combine import combine_reports
import matplotlib.pyplot as plt
from heuristic_policy import simple_policy

env_fn = waterworld_v4
current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define configurations for different models and scenarios
model_test_configs = [
    {"model_name": "PPO", "model_path": r"models\train\waterworld_v4_20240318-174101.zip", "n_pursuers": 2},
    {"model_name": "PPO", "model_path": r"models\train\waterworld_v4_20240313-175341.zip", "n_pursuers": 4},
    {"model_name": "PPO", "model_path": r"models\train\waterworld_v4_20240314-005719.zip", "n_pursuers": 6},
    {"model_name": "PPO", "model_path": r"models\train\waterworld_v4_20240314-005719.zip", "n_pursuers": 6, "sensor_range": 0.04, "poison_speed": 0.15},
    {"model_name": "PPO", "model_path": r"models\train\waterworld_v4_20240405-124706.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
    {"model_name": "SAC", "model_path": r"models\train\waterworld_v4_20240314-220608.zip", "n_pursuers": 2},
    {"model_name": "SAC", "model_path": r"models\train\waterworld_v4_20240318-022358.zip", "n_pursuers": 4},
    {"model_name": "SAC", "model_path": r"models\train\waterworld_v4_20240315-145042.zip", "n_pursuers": 6},
    {"model_name": "SAC", "model_path": r"models\train\waterworld_v4_20240406-120159.zip", "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
    {"model_name": "Heuristic", "model_path": None, "n_pursuers": 4},
    {"model_name": "Heuristic", "model_path": None, "n_pursuers": 6},
    {"model_name": "Heuristic", "model_path": None, "n_pursuers": 8, "sensor_range": 0.04, "poison_speed": 0.15, "sensor_count": 8},
]

def eval_with_model_path_run(env_fn, model_path, model_name, num_pursuers, sensor_range=None, poison_speed=None, sensor_count=None, num_games=100, render_mode=None):
    # Dynamic environment configuration based on the model specifics
    env_kwargs = {
        "n_pursuers": num_pursuers,
        "n_evaders": 6,
        "n_poisons": 8,
        "n_coop": 2,
        "n_sensors": sensor_count if sensor_count is not None else 16,
        "sensor_range": sensor_range if sensor_range is not None else 0.2,
        "radius": 0.015,
        "obstacle_radius": 0.055,
        "n_obstacles": 1,
        "obstacle_coord": [(0.5, 0.5)],
        "pursuer_max_accel": 0.01,
        "evader_speed": 0.01,
        "poison_speed": poison_speed if poison_speed is not None else 0.075,
        "poison_reward": -10,
        "food_reward": 70.0,
        "encounter_reward": 0.015,
        "thrust_penalty": -0.01,
        "local_ratio": 0.0,
        "speed_features": True,
        "max_cycles": 1000
    }
    actions = []
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    if model_path is None:
        if model_name == "Heuristic":
            print(f"Proceeding with heuristic policy for {num_games} games.")
        else:
            print("Model path is None but model name is not 'Heuristic'.")
            return None
    else:
        if not os.path.exists(model_path):
            print("Model not found.")
            return None
        if model_name == "PPO":
            model = PPO.load(model_path)
        elif model_name == "SAC":
            model = SAC.load(model_path)
        else:
            print("Invalid model name.")
            return None

    total_rewards = {agent: 0 for agent in env.possible_agents}
    episode_avg_rewards = []

    for i in range(num_games):
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        env.reset(seed=i)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            episode_rewards[agent] += reward

            if termination or truncation:
                action = None
            else:
                if model_name == "Heuristic":
                    n_sensors = env_kwargs.get('n_sensors')
                    action = simple_policy(obs, n_sensors, env_kwargs['sensor_range'])
                else:
                    action, _states = model.predict(obs, deterministic=True)
                    
                # # Store action, reward, and agent ID as a single array for later analysis
                # actionid = np.append(action, [reward, int(agent[-1])])  # Ensure this matches the expected input format for Analysis
                # actions.append(actionid)
                
                # Retrieve individual position from the info dictionary
                position = info.get('pursuer_position', np.array([0, 0]))  # Default to [0, 0] if no data available

                # Store action, individual position, reward, and agent ID as a single array for later analysis
                actionid = np.concatenate((action, position, [reward, int(agent[-1])]))  # Ensure this matches the expected input format for Analysis
                actions.append(actionid)
                
            env.step(action)

        for agent in episode_rewards:
            total_rewards[agent] += episode_rewards[agent]
        episode_avg_rewards.append(sum(episode_rewards.values()) / len(episode_rewards))

    env.close()

    overall_avg_reward = sum(total_rewards.values()) / (len(total_rewards) * num_games)
    print("Total Rewards: ", total_rewards, "over", num_games, "games")
    print(f"Overall Avg reward: {overall_avg_reward}")
    
    #print("Last 20 actions: ", actions[-20:])

    return {'actions': actions, 'overall_avg_reward': overall_avg_reward}

def run_and_analyze_all_configs(games=100):
    all_results = {}  

    for config in model_test_configs:
        config_suffix = "M" if "sensor_range" in config else ""
        print(" ---",
              "\n\n",
              f"Evaluating configuration: {config['model_name']} with {config['n_pursuers']} pursuers",
              "\n\n",
              "---"
              
              )
        


        # Run the model evaluation
        eval_results = eval_with_model_path_run(
            env_fn=env_fn,
            model_path=config.get("model_path"),
            model_name=config["model_name"],
            num_pursuers=config["n_pursuers"],
            sensor_range=config.get("sensor_range"),
            poison_speed=config.get("poison_speed"),
            sensor_count=config.get("sensor_count"),
            num_games=games,
            render_mode=None
        )
        if eval_results:
            data = eval_results.get('actions', [])
            avg_reward = eval_results.get('overall_avg_reward', 'Data Not Available')

            
            config_key = f"{config['model_name']}_pursuers_{config['n_pursuers']}{config_suffix}"
            output_dir = f"results/{current_datetime}/{config_key}"
            
            analysis = Analysis(data, output_dir=output_dir)
            
            

            all_results[config_key] = {
                                            'data': data,
                                            
                                            'avg_reward': avg_reward
                                        }
            
            
            
            
            analysis.trajectory_clustering()
            analysis.agent_density_heatmap()
            analysis.analyze_rewards_correlation()
            analysis.cluster_by_reward_category()
            
            
        
        else:
            print(f"Failed to get results for {config['model_name']} with {config['n_pursuers']} pursuers.")

    combine_dir = f"results/{current_datetime}"
    combine_reports(combine_dir)
    
    
    # with open('pickles/all_results.pkl', 'wb') as file:
    #             pickle.dump(all_results, file)
    
    
    
    return all_results

if __name__ == "__main__":
    evals = 1
    
    output_dir = f"results/{current_datetime}_{evals}evals"
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = run_and_analyze_all_configs(games=1)
