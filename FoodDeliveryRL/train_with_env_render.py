import torch
import numpy as np
from pathlib import Path

from environment.food_delivery_env import FoodDeliveryGridEnvironment
from models.dqn_agent import DQNAgent
from training.dqn_trainer import DQNTrainer

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    """Train with environment visualization"""
    
    print("\n" + "=" * 70)
    print("Food Delivery RL: Training WITH Environment Visualization")
    print("=" * 70)
    
    # Setup environment
    print("\n[STEP 1] Setting up environment...")
    env = FoodDeliveryGridEnvironment(
        grid_size=10,
        num_restaurants=3,
        num_customers=5
    )
    print(f"✓ Environment created")
    
    # Initialize agent
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
    print(f"✓ Agent created (Device: {config['device']})")
    
    # Train with rendering
    print("\n[STEP 3] Training with environment rendering...")
    log_dir = Path("logs/rendering")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = DQNTrainer(env, agent, str(log_dir), enable_plotting=False)
    
    stats = trainer.train_with_rendering(
        num_episodes=50,           # Number of training episodes
        render_interval=2,         # Render every 2 episodes
        save_video=True            # Create MP4 video
    )
    
    trainer.save_training_log("training_log.json")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Best Reward: {stats['best_reward']:.2f}")
    print(f"✓ Avg Reward: {stats['avg_reward']:.2f}")
    print(f"✓ Rendered frames saved to: ./logs/rendering/")
    print(f"✓ Video saved to: ./logs/rendering/training_video.mp4")
    print(f"✓ Models saved to: ./logs/rendering/")

if __name__ == "__main__":
    main()
