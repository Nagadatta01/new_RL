import numpy as np
import pandas as pd
from itertools import product
from typing import Dict
import sys
sys.path.append('..')

from models.dqn_agent import DQNAgent
from environment.food_delivery_env import FoodDeliveryGridEnvironment
from training.dqn_trainer import DQNTrainer

class HyperparameterTuner:
    """Hyperparameter Tuning using Grid Search"""
    
    def __init__(self, log_dir: str = "logs/tuning"):
        """Initialize tuner"""
        self.log_dir = log_dir
        self.results = []
    
    def grid_search(self,
                    param_grid: Dict,
                    env_config: Dict,
                    num_trials: int = 3,
                    train_episodes: int = 200) -> pd.DataFrame:
        """
        Perform grid search over hyperparameters
        
        Args:
            param_grid: Dictionary of parameter ranges
            env_config: Environment configuration
            num_trials: Number of independent trials per config
            train_episodes: Episodes per training run
            
        Returns:
            Results dataframe
        """
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        configs = list(product(*param_values))
        print(f"Grid Search: {len(configs)} configurations, "
              f"{num_trials} trials each = {len(configs) * num_trials} runs\n")
        
        for config_idx, config in enumerate(configs):
            config_dict = dict(zip(param_names, config))
            
            trial_rewards = []
            trial_success_rates = []
            
            for trial in range(num_trials):
                # Create environment and agent
                env = FoodDeliveryGridEnvironment(**env_config)
                agent = DQNAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_space.n,
                    **config_dict
                )
                
                # Train
                trainer = DQNTrainer(env, agent)
                trainer.train(num_episodes=train_episodes)
                
                # Evaluate
                eval_stats = trainer.evaluate(num_episodes=30)
                
                trial_rewards.append(eval_stats["avg_reward"])
                trial_success_rates.append(eval_stats["success_rate"])
            
            # Store results
            result = {
                **config_dict,
                "avg_reward_mean": np.mean(trial_rewards),
                "avg_reward_std": np.std(trial_rewards),
                "success_rate_mean": np.mean(trial_success_rates),
                "success_rate_std": np.std(trial_success_rates)
            }
            
            self.results.append(result)
            
            print(f"Config {config_idx + 1}/{len(configs)}: "
                  f"Reward: {result['avg_reward_mean']:.2f}±{result['avg_reward_std']:.2f}, "
                  f"Success: {result['success_rate_mean']:.2%}")
        
        results_df = pd.DataFrame(self.results)
        return results_df
