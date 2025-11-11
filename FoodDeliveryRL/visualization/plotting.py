import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict

class TrainingVisualizer:
    """Visualization for training metrics"""
    
    @staticmethod
    def plot_training_curves(
        episode_rewards: List[float],
        episode_losses: List[float],
        output_path: str = "training_curves.png"):
        """Plot training curves"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reward curve
        axes[0].plot(episode_rewards, alpha=0.6, label="Episode Reward")
        
        # Moving average
        window = 20
        moving_avg = np.convolve(
            episode_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[0].plot(moving_avg, label=f"Moving Avg (window={window})", linewidth=2)
        
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Rewards")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss curve
        if episode_losses:
            axes[1].plot(episode_losses, alpha=0.6, color='orange')
            axes[1].set_xlabel("Training Step")
            axes[1].set_ylabel("Loss (MSE)")
            axes[1].set_title("Training Loss")
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {output_path}")
    
    @staticmethod
    def plot_exploration_schedule(
        epsilon_values: List[float],
        output_path: str = "exploration_schedule.png"):
        """Plot exploration rate decay"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epsilon_values, linewidth=2, color='green')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon (Exploration Rate)")
        ax.set_title("ε-Greedy Exploration Schedule")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Exploration schedule saved to {output_path}")
    
    @staticmethod
    def plot_comparison(
        baseline_rewards: List[float],
        tuned_rewards: List[float],
        output_path: str = "model_comparison.png"):
        """Compare baseline vs tuned model"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(len(baseline_rewards))
        window = 20
        
        baseline_ma = np.convolve(
            baseline_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        tuned_ma = np.convolve(
            tuned_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        
        ax.plot(baseline_ma, label="Baseline DQN", linewidth=2)
        ax.plot(tuned_ma, label="Tuned DQN", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (Moving Average)")
        ax.set_title("Baseline vs Tuned DQN Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {output_path}")
