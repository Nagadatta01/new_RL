import numpy as np
import pandas as pd
from typing import Dict, List
import sys
sys.path.append('..')

class PerformanceEvaluator:
    """Comprehensive Performance Evaluation"""
    
    @staticmethod
    def compute_metrics(agent,
                       env,
                       num_episodes: int = 50) -> Dict:
        """
        Compute comprehensive metrics
        
        Args:
            agent: Trained agent
            env: Environment
            num_episodes: Evaluation episodes
            
        Returns:
            Dictionary of metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        delivery_counts = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            step_count = 0
            delivery_count = 0
            
            while step_count < env.max_steps:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                if "delivery" in info:
                    delivery_count += 1
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            delivery_counts.append(delivery_count)
            
            # Check if all orders delivered
            if all(order["delivered"] for order in env.orders.values()):
                success_count += 1
        
        return {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes,
            "avg_deliveries": np.mean(delivery_counts),
            "convergence_speed": episode_lengths[0] - episode_lengths[-1]
        }
