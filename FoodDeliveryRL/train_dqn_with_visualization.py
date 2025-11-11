import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    print("\n" + "="*80)
    print("DQN TRAINING WITH LIVE VISUALIZATION")
    print("Intelligent Autonomous Vehicle for Food Delivery")
    print("="*80)
    
    # Setup
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    
    print(f"\n✓ Environment Setup:")
    print(f"  - Grid Size: 15×15")
    print(f"  - Restaurants: 3")
    print(f"  - Customers: 5 orders per episode")
    print(f"  - State Space: {env.state_dim} dimensions")
    print(f"  - Action Space: 6 discrete actions")
    
    # DQN Agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\n✓ DQN Agent Created")
    print(f"  - Learning Rate: 0.001")
    print(f"  - Gamma: 0.95")
    print(f"  - Epsilon Decay: 0.995")
    print(f"  - Device: {str(agent.device).upper()}")

    
    # Training parameters
    num_episodes = 300
    render_every = 5  # Render every 5 episodes
    save_interval = 50
    
    log_dir = Path("logs/dqn_visual")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*80)
    print(f"TRAINING STARTED")
    print(f"="*80)
    print(f"\n✓ Total Episodes: {num_episodes}")
    print(f"✓ Visualization: Every {render_every} episodes")
    print(f"✓ Logs saved to: {log_dir}/\n")
    
    # Training loop
    rewards = []
    deliveries = []
    best_reward = -np.inf
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        # Enable rendering every N episodes
        render_episode = (episode + 1) % render_every == 0
        
        while step_count < env.max_steps:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Render if enabled for this episode
            if render_episode:
                env.render()
            
            if done:
                break
        
        # Update epsilon
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        rewards.append(episode_reward)
        deliveries.append(env.deliveries)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(str(log_dir / "best_dqn_model.pt"))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_del = np.mean(deliveries[-10:])
            eps = agent.epsilon if hasattr(agent, 'epsilon') else 0
            print(f"Episode {episode+1:3d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Deliveries: {avg_del:.2f}/5 | "
                  f"Epsilon: {eps:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save_model(str(log_dir / f"dqn_checkpoint_ep{episode+1}.pt"))
    
    # Close rendering
    env.close()
    
    # Save final model
    agent.save_model(str(log_dir / "final_dqn_model.pt"))
    
    # Plot training progress
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    window = 20
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(np.convolve(rewards, np.ones(window)/window, mode='valid'), 
             label=f'{window}-Episode Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(deliveries, alpha=0.3, label='Raw')
    plt.plot(np.convolve(deliveries, np.ones(window)/window, mode='valid'), 
             label=f'{window}-Episode Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Deliveries')
    plt.title('Successful Deliveries (out of 5)')
    plt.axhline(y=5, color='green', linestyle='--', label='Perfect (5/5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / 'training_progress.png', dpi=150)
    print(f"\n✓ Training plot saved: {log_dir}/training_progress.png")
    
    # Save training data
    training_data = {
        "episodes": list(range(1, num_episodes + 1)),
        "rewards": [float(r) for r in rewards],
        "deliveries": [int(d) for d in deliveries]
    }
    
    with open(log_dir / "training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    # Final summary
    print(f"\n" + "="*80)
    print("TRAINING COMPLETED")
    print(f"="*80)
    
    print(f"\nFinal Results (Last 50 episodes):")
    print(f"  - Average Reward: {np.mean(rewards[-50:]):.2f}")
    print(f"  - Average Deliveries: {np.mean(deliveries[-50:]):.2f}/5")
    print(f"  - Best Episode Reward: {best_reward:.2f}")
    print(f"  - Final Epsilon: {agent.epsilon if hasattr(agent, 'epsilon') else 'N/A':.4f}")
    
    print(f"\n✓ Models saved:")
    print(f"  - Best model: {log_dir}/best_dqn_model.pt")
    print(f"  - Final model: {log_dir}/final_dqn_model.pt")
    print(f"  - Checkpoints: {log_dir}/dqn_checkpoint_ep*.pt")
    
    print(f"\n✓ Data saved:")
    print(f"  - Training data: {log_dir}/training_data.json")
    print(f"  - Progress plot: {log_dir}/training_progress.png")
    
    print(f"\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
