import torch
import numpy as np
from pathlib import Path

from environment.food_delivery_env import FoodDeliveryGridEnvironment
from models.dqn_agent import DQNAgent
from training.dqn_trainer import DQNTrainer

# Set seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    """Train with live visualization using Gymnasium"""
    
    print("\n" + "=" * 70)
    print("Food Delivery RL: Live Training Visualization (Gymnasium)")
    print("=" * 70)
    
    # Environment
    print("\n[STEP 1] Setting up Gymnasium environment...")
    env = FoodDeliveryGridEnvironment(
        grid_size=10,
        num_restaurants=3,
        num_customers=5
    )
    print(f"✓ Gymnasium environment ready")
    
    # Agent
    print("\n[STEP 2] Initializing DQN Agent...")
    config = {
        "state_dim": env.state_dim,
        "action_dim": env.action_space.n,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 100000,
        "batch_size": 32,
        "target_update_freq": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    agent = DQNAgent(**config)
    print(f"✓ DQN Agent created (Device: {config['device']})")
    
    # Training with live visualization
    print("\n[STEP 3] Starting live training visualization...")
    log_dir = Path("logs/live_viz")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = DQNTrainer(env, agent, str(log_dir))
    
    stats = trainer.train_live_visualization(
        num_episodes=100,
        update_interval=1  # Update visualization every episode
    )
    
    trainer.save_training_log("training_log.json")
    
    print(f"\n✓ Best Reward: {stats['best_reward']:.2f}")
    print(f"✓ Avg Reward: {stats['avg_reward']:.2f}")
    print(f"✓ Models saved to: ./logs/live_viz/")

if __name__ == "__main__":
    main()
