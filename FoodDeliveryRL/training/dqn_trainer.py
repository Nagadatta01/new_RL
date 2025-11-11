import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import torch

class DQNTrainer:
    """DQN Training Pipeline"""
    
    def __init__(self, env, agent, log_dir: str = "logs"):
        self.env = env
        self.agent = agent
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_losses = []
        self.exploration_rates = []
    
    def train_episode(self) -> Tuple[float, float]:
        """Train for one episode"""
        state, info = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        while True:
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.agent.store_experience(state, action, reward, next_state, done)
            loss = self.agent.train_step()
            
            episode_reward += reward
            episode_loss += loss
            step_count += 1
            
            state = next_state
            
            if done:
                break
        
        avg_loss = episode_loss / max(step_count, 1)
        
        self.agent.epsilon = max(self.agent.epsilon_end, self.agent.epsilon * self.agent.epsilon_decay)
        
        self.episode_rewards.append(episode_reward)
        self.episode_losses.append(avg_loss)
        self.exploration_rates.append(self.agent.epsilon)
        
        return episode_reward, avg_loss
    
    def train(self, num_episodes: int = 500) -> Dict:
        """Train agent"""
        best_reward = -np.inf
        
        print("\n" + "="*70)
        print("DQN TRAINING STARTED")
        print("="*70)
        
        for episode in range(num_episodes):
            episode_reward, avg_loss = self.train_episode()
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:>8.2f} | Epsilon: {self.agent.epsilon:.4f}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save_model(str(self.log_dir / "best_model.pt"))
        
        self.agent.save_model(str(self.log_dir / "final_model.pt"))
        
        print("="*70 + "\n")
        
        return self._get_training_stats()
    
    def evaluate(self, num_episodes: int = 50, noise: float = 0.0) -> Dict:
        """Evaluate trained agent"""
        eval_rewards = []
        success_count = 0
        
        for ep in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            while True:
                if noise > 0:
                    noisy_state = state + np.random.normal(0, noise, state.shape)
                    noisy_state = np.clip(noisy_state, 0, self.env.grid_size)
                else:
                    noisy_state = state
                
                action = self.agent.select_action(noisy_state, training=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                step_count += 1
                state = next_state
                
                all_delivered = all(order["delivered"] for order in self.env.orders.values())
                
                if done:
                    if all_delivered:
                        success_count += 1
                    break
            
            eval_rewards.append(episode_reward)
        
        return {
            "avg_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "success_rate": success_count / num_episodes,
            "all_rewards": eval_rewards
        }
    
    def _get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            "best_reward": np.max(self.episode_rewards),
            "final_epsilon": self.agent.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_losses": self.episode_losses,
            "exploration_rates": self.exploration_rates
        }
