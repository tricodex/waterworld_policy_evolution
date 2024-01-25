# heuristic_policy.py

from pettingzoo.sisl import waterworld_v4
import numpy as np
import matplotlib.pyplot as plt
from settings import env_kwargs

env = waterworld_v4.env(**env_kwargs, render_mode=None) #  human, rgb_array, ansi
env.reset(seed=42)

def simple_policy(observation, n_sensors, sensor_range):
    # Extract sensor readings for food and poison
    food_dist = observation[2 * n_sensors:3 * n_sensors]
    poison_dist = observation[4 * n_sensors:5 * n_sensors]

    # Threshold to consider an object "detected"
    detection_threshold = 0.5 * sensor_range

    # Initialize action
    action = np.array([0.0, 0.0])

    # Check for food and poison
    if np.any(food_dist < detection_threshold):
        # Move towards the closest food
        closest_food_index = np.argmin(food_dist)
        action[0] = np.cos(closest_food_index * 2 * np.pi / n_sensors)
        action[1] = np.sin(closest_food_index * 2 * np.pi / n_sensors)
    elif np.any(poison_dist < detection_threshold):
        # Move away from the closest poison
        closest_poison_index = np.argmin(poison_dist)
        action[0] = -np.cos(closest_poison_index * 2 * np.pi / n_sensors)
        action[1] = -np.sin(closest_poison_index * 2 * np.pi / n_sensors)
    else:
        # Random wander
        random_direction = np.random.rand() * 2 * np.pi
        action[0] = np.cos(random_direction)
        action[1] = np.sin(random_direction)

    # Assuming 'action' is the variable holding your calculated action
    action = np.array(action, dtype=np.float32)


    return action

# Main simulation loop
n_sensors = 30  # Assuming 30 sensors as per the default environment setup
sensor_range = 0.2  # Assuming default sensor range

# Initialize reward tracking
total_rewards = {agent: 0 for agent in env.agents}

# Main simulation loop
for agent in env.agent_iter():
    observation, reward, termination, truncation, _ = env.last()
    if termination or truncation:
        action = None
    else:
        action = simple_policy(observation, n_sensors, sensor_range)
    env.step(action)

    # Update total rewards
    total_rewards[agent] += reward

env.close()

# Plotting the total rewards
plt.bar(total_rewards.keys(), total_rewards.values())
plt.xlabel('Agents')
plt.ylabel('Total Rewards')
plt.title('Total Rewards per Agent in Waterworld Simulation')
plt.show()