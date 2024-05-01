# analysis.py
# action_aray is a list of lists, each list contains 5 elements in a continious manner(actionspace+reward+id): 'HorizontalThrust', 'VerticalThrust', 'XPosition', 'YPosition', 'Reward', 'AgentID'


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

class Analysis:
    def __init__(self, actions_array, output_dir):
        self.actions_array = actions_array
        self.output_dir = output_dir
        self.df = pd.DataFrame(self.actions_array, columns=['Horizontal', 'Vertical', 'XPosition', 'YPosition', 'Reward', 'AgentID'])
        
        

        
        self.scaled_features = StandardScaler().fit_transform(self.df[['Horizontal', 'Vertical']])
        self.unique_agents = self.df['AgentID'].unique()
        
        
        
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.plot_correlation_matrix()
        
        
        # Feature Engineering
        self.df['ThrustInteraction'] = self.df['Horizontal'] * self.df['Vertical']  # Adding interaction term
        self.df['Radius'] = np.sqrt(self.df['XPosition']**2 + self.df['YPosition']**2)
        self.df['Angle'] = np.arctan2(self.df['YPosition'], self.df['XPosition'])
        self.df['Velocity_X'] = self.df['XPosition'].diff()  # Difference from the previous timestep
        self.df['Velocity_Y'] = self.df['YPosition'].diff()
        
        
        self.plot_feature_correlation_matrix()
        
        # Categorize rewards
        self.df['RewardCategory'] = self.df['Reward'].apply(
            lambda x: 'Food' if x > (69/len(self.unique_agents)) else ('Poison' if x < (-9.9/len(self.unique_agents)) else 'Neutral')
        )
        
    def analyze_rewards_correlation(self):
        

        # Group by reward category to analyze different metrics
        grouped = self.df.groupby('RewardCategory').mean()


        plt.figure(figsize=(10, 5))
        sns.barplot(x=grouped.index, y=np.sqrt(grouped['Velocity_X']**2 + grouped['Velocity_Y']**2), palette='viridis')
        plt.title('Average Movement Speed by Reward Category')
        plt.ylabel('Average Movement Speed')
        plt.savefig(f"{self.output_dir}/velocity_reward_category_analysis.png")
        plt.close()
        
    def cluster_by_reward_category(self):
        for category in self.df['RewardCategory'].unique():
            cat_data = self.df[self.df['RewardCategory'] == category].copy()
            features = cat_data[['Velocity_X', 'Velocity_Y']]
            
            # Handling NaN values
            features.fillna(0, inplace=True)  # Fill NaNs with the mean of each column

            # Alternatively, to drop rows with NaNs, uncomment the following line:
            # features.dropna(inplace=True)
            
            if not features.empty:
                clustering = KMeans(n_clusters=2).fit(features)
                cat_data.loc[features.index, 'Cluster'] = clustering.labels_

                plt.figure()
                sns.scatterplot(x='Velocity_X', y='Velocity_Y', hue='Cluster', data=cat_data)
                plt.title(f'Clustering of Movements in {category}')
                plt.savefig(f"{self.output_dir}/cluster_{category}.png")
                plt.close()

            
    def plot_feature_correlation_matrix(self):
        correlation_matrix = self.df[['Horizontal', 'Vertical', 'XPosition', 
                                      'YPosition', 'Reward', 'ThrustInteraction', 'Radius', 
                                      'Angle', 'Velocity_X', 'Velocity_Y']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix Heatmap')
        plt.savefig(f"{self.output_dir}/feature_correlation_matrix.png")
        plt.close()
    
    def plot_correlation_matrix(self):
        correlation_matrix = self.df[['Horizontal', 'Vertical', 'XPosition', 'YPosition', 'Reward']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix Heatmap')
        plt.savefig(f"{self.output_dir}/correlation_matrix.png")
        plt.close()

    def trajectory_clustering(self, stop_on_high_reward=True):
        # Normalize the position data
        scaled_positions = StandardScaler().fit_transform(self.df[['XPosition', 'YPosition']])
        # Use DBSCAN to identify clusters based on spatial proximity
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(scaled_positions)
        self.df['Cluster'] = clustering.labels_

        # Calculate the reward threshold
        unique_ids = self.df['AgentID'].unique()
        reward_threshold = (70 / len(unique_ids)) * 0.9

        # Create a color palette with seaborn that has as many colors as there are unique agent IDs
        palette = sns.color_palette("hsv", len(unique_ids))
        color_map = dict(zip(unique_ids, palette))

        # Plotting
        plt.figure(figsize=(10, 8))
        for agent_id in unique_ids:
            agent_data = self.df[self.df['AgentID'] == agent_id]

            if stop_on_high_reward:
                # Identify the first index where the reward exceeds the threshold
                high_reward_index = agent_data[agent_data['Reward'] > reward_threshold].index.min()
                if pd.notna(high_reward_index):
                    # Truncate the data at the first high reward occurrence
                    agent_data = agent_data.loc[:high_reward_index]

            plt.scatter(agent_data['XPosition'], agent_data['YPosition'], 
                        color=color_map[agent_id], label=f'Agent {int(agent_id+1)}', s=50, alpha=0.7)

        plt.title('Trajectory Clustering')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(title='Agent ID')
        plt.savefig(f"{self.output_dir}/trajectory_clustering.png")
        plt.close()

    def agent_density_heatmap(self):
        # Create a grid of the environment
        x_edges = np.linspace(self.df['XPosition'].min(), self.df['XPosition'].max(), num=50)
        y_edges = np.linspace(self.df['YPosition'].min(), self.df['YPosition'].max(), num=50)
        heatmap, _, _ = np.histogram2d(self.df['XPosition'], self.df['YPosition'], bins=[x_edges, y_edges])
        
        # Plotting
        plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Agent Density Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(f"{self.output_dir}/agent_density_heatmap.png")
        plt.close()
