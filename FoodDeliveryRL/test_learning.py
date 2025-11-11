import torch
import numpy as np
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def quick_test():
    """Quick test to verify agent learning"""
    
    print("\n" + "="*70)
    print("QUICK LEARNING VERIFICATION TEST")
    print("="*70)
    
    # Setup
    env = RealisticDeliveryEnvironment(grid_size=20, num_restaurants=4, num_customers=8)
    
    config = {
        "state_dim": env.state_dim,
        "action_dim": env.action_space.n,
        "learning_rate": 0.0005,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.20,
        "epsilon_decay": 0.9995,
        "buffer_size": 50000,
        "batch_size": 16,
        "target_update_freq": 500,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    agent = DQNAgent(**config)
    
    print(f"\n✓ Environment: 20x20 Grid")
    print(f"✓ Agent: DQN with {config['state_dim']} state dims")
    print(f"✓ Device: {config['device'].upper()}")
    
    # Quick 50-episode test
    print(f"\n[Running 50 episodes for quick check...]")
    
    rewards = []
    deliveries = []
    epsilon_vals = []
    
    for episode in range(50):
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
        
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        rewards.append(episode_reward)
        deliveries.append(env.agent1_deliveries + env.agent2_deliveries)
        epsilon_vals.append(agent.epsilon)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            total_del = sum(deliveries[-10:])
            print(f"Episode {episode+1:2d} | Avg Reward (last 10): {avg_reward:7.2f} | Deliveries: {total_del}/80 | Epsilon: {agent.epsilon:.4f}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    first_10_avg = np.mean(rewards[:10])
    last_10_avg = np.mean(rewards[-10:])
    improvement = last_10_avg - first_10_avg
    
    print(f"\nReward Trend:")
    print(f"  - First 10 episodes avg: {first_10_avg:7.2f}")
    print(f"  - Last 10 episodes avg: {last_10_avg:7.2f}")
    print(f"  - Improvement: {improvement:7.2f} {'✓ LEARNING!' if improvement > 0 else '✗ NOT LEARNING'}")
    
    first_del = sum(deliveries[:10])
    last_del = sum(deliveries[-10:])
    
    print(f"\nDelivery Count:")
    print(f"  - First 10 episodes: {first_del}/80 orders delivered")
    print(f"  - Last 10 episodes: {last_del}/80 orders delivered")
    print(f"  - Improvement: {last_del - first_del:+d} {'✓ LEARNING!' if last_del > first_del else '✗ NOT LEARNING'}")
    
    print(f"\nExploration:")
    print(f"  - Start epsilon: {epsilon_vals[0]:.4f}")
    print(f"  - End epsilon: {epsilon_vals[-1]:.4f}")
    print(f"  - Decay: {'✓ SMOOTH' if abs(epsilon_vals[-1] - 0.20) < 0.02 else '✗ NOT SMOOTH'}")
    
    if improvement > 0 and last_del > first_del:
        print(f"\n{'='*70}")
        print("✓ AGENT IS LEARNING! Proceed with full training")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"\n{'='*70}")
        print("✗ AGENT NOT LEARNING. Check reward function!")
        print(f"{'='*70}\n")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
