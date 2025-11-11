import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent
from models.ppo_agent import PPOAgent
from models.a2c_agent import A2CAgent

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_agent(env, agent, agent_name: str, num_episodes: int = 300):
    """Train single agent and return results"""
    print(f"\n{'='*70}")
    print(f"Training {agent_name}")
    print(f"{'='*70}")
    
    rewards = []
    deliveries = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < env.max_steps:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        rewards.append(episode_reward)
        deliveries.append(env.deliveries)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards[-50:])
            avg_del = np.mean(deliveries[-50:])
            print(f"Episode {episode+1:3d} | Avg Reward: {avg_reward:7.2f} | Avg Deliveries: {avg_del:.2f}/5")
    
    return rewards, deliveries

def main():
    print("\n" + "="*80)
    print("COMPARING RL ALGORITHMS FOR FOOD DELIVERY")
    print("DQN vs PPO vs A2C")
    print("="*80)
    
    # Environment
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    
    print(f"\n✓ Environment Setup:")
    print(f"  - Grid: 15×15")
    print(f"  - Restaurants: 3")
    print(f"  - Customers: 5 orders per episode")
    print(f"  - State Space: {env.state_dim} dims (simplified)")
    print(f"  - Action Space: 6 discrete actions")
    
    # Create agents
    dqn_agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100
    )
    
    ppo_agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_space.n, lr=0.0003)
    a2c_agent = A2CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, lr=0.001)
    
    # Train all agents
    num_episodes = 300
    
    print(f"\n[Training DQN...]")
    dqn_rewards, dqn_deliveries = train_agent(env, dqn_agent, "DQN", num_episodes)
    
    print(f"\n[Training PPO...]")
    ppo_rewards, ppo_deliveries = train_agent(env, ppo_agent, "PPO", num_episodes)
    
    print(f"\n[Training A2C...]")
    a2c_rewards, a2c_deliveries = train_agent(env, a2c_agent, "A2C", num_episodes)
    
    # Plot comparison
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    window = 20
    plt.plot(np.convolve(dqn_rewards, np.ones(window)/window, mode='valid'), label='DQN', linewidth=2)
    plt.plot(np.convolve(ppo_rewards, np.ones(window)/window, mode='valid'), label='PPO', linewidth=2)
    plt.plot(np.convolve(a2c_rewards, np.ones(window)/window, mode='valid'), label='A2C', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Moving Avg)')
    plt.title('Training Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.convolve(dqn_deliveries, np.ones(window)/window, mode='valid'), label='DQN', linewidth=2)
    plt.plot(np.convolve(ppo_deliveries, np.ones(window)/window, mode='valid'), label='PPO', linewidth=2)
    plt.plot(np.convolve(a2c_deliveries, np.ones(window)/window, mode='valid'), label='A2C', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Deliveries (Moving Avg)')
    plt.title('Deliveries Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/algorithm_comparison.png', dpi=150)
    print(f"\n✓ Comparison plot saved: logs/algorithm_comparison.png")
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS (Last 50 episodes)")
    print(f"{'='*80}")
    
    print(f"\nDQN:")
    print(f"  - Avg Reward: {np.mean(dqn_rewards[-50:]):.2f}")
    print(f"  - Avg Deliveries: {np.mean(dqn_deliveries[-50:]):.2f}/5")
    
    print(f"\nPPO:")
    print(f"  - Avg Reward: {np.mean(ppo_rewards[-50:]):.2f}")
    print(f"  - Avg Deliveries: {np.mean(ppo_deliveries[-50:]):.2f}/5")
    
    print(f"\nA2C:")
    print(f"  - Avg Reward: {np.mean(a2c_rewards[-50:]):.2f}")
    print(f"  - Avg Deliveries: {np.mean(a2c_deliveries[-50:]):.2f}/5")
    
    # Save results
    results = {
        "dqn": {"rewards": [float(r) for r in dqn_rewards], "deliveries": [int(d) for d in dqn_deliveries]},
        "ppo": {"rewards": [float(r) for r in ppo_rewards], "deliveries": [int(d) for d in ppo_deliveries]},
        "a2c": {"rewards": [float(r) for r in a2c_rewards], "deliveries": [int(d) for d in a2c_deliveries]}
    }
    
    Path("logs").mkdir(exist_ok=True)
    with open("logs/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: logs/comparison_results.json")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
