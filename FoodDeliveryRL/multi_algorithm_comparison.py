"""
Multi-Algorithm Comparison - 3 Deep RL Algorithms
Trains and compares: DQN, Double DQN, and A2C (No Random Baseline)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent
from models.double_dqn_agent import DoubleDQNAgent
from models.a2c_agent import A2CAgent


def train_agent(agent_class, agent_name, num_episodes=200, num_runs=3):
    """Train agent and return results"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {agent_name}")
    print(f"{'='*80}")
    
    all_runs_rewards = []
    all_runs_deliveries = []
    
    for run in range(num_runs):
        print(f"\n  Run {run + 1}/{num_runs}...")
        
        env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
        
        # Create agent
        agent = agent_class(
            state_dim=5,
            action_dim=6,
            learning_rate=0.001,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32,
            target_update=100
        )
        
        episode_rewards = []
        episode_deliveries = []
        
        for episode in range(num_episodes):
            state, _ = env.reset(seed=42 + run * 1000 + episode)
            episode_reward = 0
            
            for step in range(200):
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done or truncated)
                agent.train_step()
                
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_deliveries.append(info['deliveries'])
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_del = np.mean(episode_deliveries[-50:])
                print(f"    Ep {episode + 1:3d} | Reward: {avg_reward:7.2f} | Deliveries: {avg_del:.2f}/5")
        
        all_runs_rewards.append(episode_rewards)
        all_runs_deliveries.append(episode_deliveries)
        env.close()
    
    # Calculate statistics
    all_rewards_array = np.array(all_runs_rewards)
    final_rewards = all_rewards_array[:, -50:].mean(axis=1)  # Last 50 episodes
    
    all_deliveries_array = np.array(all_runs_deliveries)
    final_deliveries = all_deliveries_array[:, -50:].mean(axis=1)
    
    result = {
        "agent_name": agent_name,
        "all_runs_rewards": all_runs_rewards,
        "all_runs_deliveries": all_runs_deliveries,
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "mean_deliveries": float(np.mean(final_deliveries)),
        "std_deliveries": float(np.std(final_deliveries))
    }
    
    print(f"\n  ✓ {agent_name} Final Performance:")
    print(f"    Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    print(f"    Deliveries: {result['mean_deliveries']:.2f} ± {result['std_deliveries']:.2f}")
    
    return result


def multi_algorithm_comparison():
    """Run comparison of 3 deep RL algorithms"""
    
    print("\n" + "="*80)
    print("MULTI-ALGORITHM COMPARISON - DEEP RL METHODS")
    print("="*80)
    print("\nTraining 3 Deep RL Algorithms:")
    print("  1. DQN (Standard Deep Q-Network)")
    print("  2. Double DQN (Reduced Overestimation)")
    print("  3. A2C (Advantage Actor-Critic)")
    print("\nEach algorithm: 3 runs × 200 episodes = 600 episodes")
    print("Total training: 1,800 episodes\n")
    
    # Train all agents
    results = []
    
    # 1. DQN
    results.append(train_agent(DQNAgent, "DQN", num_episodes=200, num_runs=3))
    
    # 2. Double DQN (expected best)
    results.append(train_agent(DoubleDQNAgent, "Double DQN", num_episodes=200, num_runs=3))
    
    # 3. A2C
    results.append(train_agent(A2CAgent, "A2C", num_episodes=200, num_runs=3))
    
    # ===== GENERATE COMPARISON PLOTS =====
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS...")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Learning Curves
    ax1 = plt.subplot(2, 3, 1)
    colors = ['blue', 'green', 'orange']
    for i, result in enumerate(results):
        # Average across runs
        avg_rewards = np.mean(result['all_runs_rewards'], axis=0)
        # Smooth
        window = 20
        smoothed = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=result['agent_name'], color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward (Smoothed)', fontsize=12)
    ax1.set_title('Learning Curves Comparison', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 2: Final Performance (Rewards)
    ax2 = plt.subplot(2, 3, 2)
    agent_names = [r['agent_name'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    
    bars = ax2.bar(range(len(results)), mean_rewards, yerr=std_rewards,
                   capsize=5, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(agent_names, fontsize=11)
    ax2.set_ylabel('Final Avg Reward (Last 50 eps)', fontsize=12)
    ax2.set_title('Final Performance Comparison', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Plot 3: Delivery Performance
    ax3 = plt.subplot(2, 3, 3)
    mean_deliveries = [r['mean_deliveries'] for r in results]
    std_deliveries = [r['std_deliveries'] for r in results]
    
    bars = ax3.bar(range(len(results)), mean_deliveries, yerr=std_deliveries,
                   capsize=5, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels(agent_names, fontsize=11)
    ax3.set_ylabel('Avg Deliveries (Last 50 eps)', fontsize=12)
    ax3.set_title('Delivery Success Comparison', fontsize=14, weight='bold')
    ax3.set_ylim(0, 5)
    ax3.axhline(y=5, color='green', linestyle='--', linewidth=1, label='Target')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Plot 4: Relative Performance
    ax4 = plt.subplot(2, 3, 4)
    baseline_reward = min([r['mean_reward'] for r in results])  # Worst performer as baseline
    improvements = [(r['mean_reward'] - baseline_reward) for r in results]
    
    bars = ax4.barh(range(len(results)), improvements, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(results)))
    ax4.set_yticklabels(agent_names, fontsize=11)
    ax4.set_xlabel('Reward Difference from Worst', fontsize=12)
    ax4.set_title('Relative Performance', fontsize=14, weight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'+{width:.1f}', ha='left', va='center', fontsize=10)
    
    # Plot 5: Convergence Speed
    ax5 = plt.subplot(2, 3, 5)
    convergence_episodes = []
    threshold = 300  # Reward threshold for "convergence"
    
    for result in results:
        avg_rewards = np.mean(result['all_runs_rewards'], axis=0)
        window = 20
        smoothed = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
        
        # Find first episode where smoothed reward > threshold
        converged = np.where(smoothed > threshold)[0]
        if len(converged) > 0:
            convergence_episodes.append(converged[0])
        else:
            convergence_episodes.append(200)  # Didn't converge
    
    bars = ax5.bar(range(len(results)), convergence_episodes, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(results)))
    ax5.set_xticklabels(agent_names, fontsize=11)
    ax5.set_ylabel('Episodes to Convergence', fontsize=12)
    ax5.set_title(f'Convergence Speed (Reward > {threshold})', fontsize=14, weight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 220)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Plot 6: Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = [["Algorithm", "Reward", "Deliveries", "Rank"]]
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    for i, result in enumerate(sorted_results):
        table_data.append([
            result['agent_name'],
            f"{result['mean_reward']:.1f} ± {result['std_reward']:.1f}",
            f"{result['mean_deliveries']:.2f}",
            f"#{i+1}"
        ])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.3, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color winner row
    table[(1, 0)].set_facecolor('#FFD700')
    table[(1, 1)].set_facecolor('#FFD700')
    table[(1, 2)].set_facecolor('#FFD700')
    table[(1, 3)].set_facecolor('#FFD700')
    
    ax6.set_title('Performance Summary', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    os.makedirs("results/multi_algorithm", exist_ok=True)
    plt.savefig("results/multi_algorithm/algorithm_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: results/multi_algorithm/algorithm_comparison.png")
    
    # Save results JSON
    results_summary = {
        "algorithms": [
            {
                "name": r['agent_name'],
                "mean_reward": r['mean_reward'],
                "std_reward": r['std_reward'],
                "mean_deliveries": r['mean_deliveries'],
                "std_deliveries": r['std_deliveries']
            }
            for r in results
        ]
    }
    
    with open("results/multi_algorithm/comparison_results.json", 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"✓ Results saved to: results/multi_algorithm/comparison_results.json")
    
    # Print summary
    print("\n" + "="*80)
    print("MULTI-ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by performance
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Algorithm':<18} {'Reward':<20} {'Deliveries'}")
    print("-" * 70)
    for i, result in enumerate(sorted_results):
        print(f"{i+1:<6} {result['agent_name']:<18} "
              f"{result['mean_reward']:>7.2f} ± {result['std_reward']:>5.2f}  "
              f"{result['mean_deliveries']:>4.2f} ± {result['std_deliveries']:>4.2f}")
    
    best = sorted_results[0]
    worst = sorted_results[-1]
    improvement = ((best['mean_reward'] - worst['mean_reward']) / abs(worst['mean_reward']) * 100) if worst['mean_reward'] != 0 else 0
    
    print(f"\n🏆 WINNER: {best['agent_name']}")
    print(f"   • Performance: {best['mean_reward']:.2f} reward, {best['mean_deliveries']:.2f} deliveries")
    print(f"   • Improvement over worst: {improvement:+.1f}%")
    
    print("\n" + "="*80)
    print("✅ MULTI-ALGORITHM COMPARISON COMPLETE!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    multi_algorithm_comparison()
