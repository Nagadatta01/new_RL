import torch
import numpy as np
from pathlib import Path

from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent
from visualization.slow_live_viz import SlowLiveVisualizer

# Set seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_with_slow_viz():
    """Train on realistic environment with slow visualization"""
    
    print("\n" + "=" * 70)
    print("🌆 Realistic Food Delivery: Slow-Motion Live Training")
    print("=" * 70)
    
    # Create realistic environment
    print("\n[STEP 1] Creating realistic city environment...")
    env = RealisticDeliveryEnvironment(
        grid_size=20,
        num_restaurants=4,
        num_customers=8
    )
    print("✓ Realistic environment with roads, parks, buildings created!")
    print(f"  - Grid: 20x20 (city layout)")
    print(f"  - Restaurants: 4")
    print(f"  - Customers: 8")
    print(f"  - Features: Roads, Parks, Buildings, Grass")
    
    # Create agent
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
    print(f"✓ Agent ready (Device: {config['device']})")
    
    # Create slow visualizer
    print("\n[STEP 3] Starting slow-motion training visualization...")
    print("⏱️  Visualization speed: 0.5 seconds per step")
    print("👀 Watch agents navigate around buildings!\n")
    
    viz = SlowLiveVisualizer(grid_size=20, pause_duration=0.5)
    
    log_dir = Path("logs/realistic")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    num_episodes = 20  # Fewer episodes since visualization is slow
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while step_count < env.max_steps:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store and train
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Update visualization EVERY STEP (slow motion)
            viz.update_environment(env, episode + 1, step_count, episode_reward)
            
            if done:
                break
        
        # Update rewards plot
        viz.update_rewards(episode + 1, episode_reward)
        
        # Save model every 5 episodes
        if (episode + 1) % 5 == 0:
            agent.save_model(str(log_dir / f"model_ep{episode+1}.pt"))
        
        print(f"✓ Episode {episode + 1} completed")
        print(f"  - Reward: {episode_reward:.2f}")
        print(f"  - Steps: {step_count}")
        print(f"  - Agent 1 Deliveries: {env.agent1_deliveries}")
        print(f"  - Agent 2 Deliveries: {env.agent2_deliveries}")
        print(f"  - Total Deliveries: {sum(1 for o in env.orders.values() if o['delivered'])}/{len(env.orders)}")
        print(f"  - Collisions: {env.total_collisions}")
    
    viz.close()
    
    # Save final model
    agent.save_model(str(log_dir / "final_model.pt"))
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Models saved to: ./logs/realistic/")

if __name__ == "__main__":
    train_with_slow_viz()
