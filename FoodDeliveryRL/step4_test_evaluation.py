"""
[STEP 4] TEST MODULE & PERFORMANCE EVALUATION (2 Marks)
Test on altered environments + noise
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42


def evaluate_scenario(name, env_config, num_episodes=10, add_noise=False, noise_level=0.1):
    """Evaluate agent on scenario"""
    
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    env = RealisticDeliveryEnvironment(**env_config)
    
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
        device="cpu"
    )
    
    metrics = {
        "rewards": [],
        "deliveries": [],
        "success": []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            if add_noise:
                state_noisy = state + np.random.normal(0, noise_level * np.abs(state).max(), state.shape)
                state_noisy = np.clip(state_noisy, 0, 15)
                action = agent.select_action(torch.tensor(state_noisy, dtype=torch.float32), training=False)
            else:
                action = agent.select_action(torch.tensor(state, dtype=torch.float32), training=False)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        success = 1 if env.deliveries == 5 else 0
        metrics["rewards"].append(episode_reward)
        metrics["deliveries"].append(env.deliveries)
        metrics["success"].append(success)
        
        print(f"  Ep {episode+1:2d} | Reward: {episode_reward:7.2f} | Deliveries: {env.deliveries}/5")
    
    env.close()
    
    print(f"\n  📊 Summary:")
    print(f"     • Avg Reward: {np.mean(metrics['rewards']):.2f}")
    print(f"     • Avg Deliveries: {np.mean(metrics['deliveries']):.2f}")
    print(f"     • Success Rate: {np.mean(metrics['success'])*100:.1f}%")
    
    return metrics


def step4_test_evaluation():
    """Step 4: Test Module and Performance Evaluation"""
    
    print("\n" + "="*80)
    print("[STEP 4] TEST MODULE & PERFORMANCE EVALUATION (2 Marks)")
    print("="*80)
    
    results_dir = Path("results/step4_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🧪 Testing on Altered Environments & Perturbations:\n")
    
    # Scenario 1: Unseen episodes
    scenario1 = evaluate_scenario(
        "TEST 1: UNSEEN EPISODES (15×15)",
        {"grid_size": 15, "num_restaurants": 3, "num_customers": 5},
        num_episodes=10,
        add_noise=False
    )
    
    # Scenario 2: Altered environment
    scenario2 = evaluate_scenario(
        "TEST 2: ALTERED ENVIRONMENT (10×10)",
        {"grid_size": 10, "num_restaurants": 3, "num_customers": 5},
        num_episodes=10,
        add_noise=False
    )
    
    # Scenario 3: Noise
    scenario3 = evaluate_scenario(
        "TEST 3: NOISY OBSERVATIONS (±10%)",
        {"grid_size": 15, "num_restaurants": 3, "num_customers": 5},
        num_episodes=10,
        add_noise=True,
        noise_level=0.1
    )
    
    # Save results
    eval_results = {
        "unseen": {
            "avg_reward": float(np.mean(scenario1["rewards"])),
            "avg_deliveries": float(np.mean(scenario1["deliveries"])),
            "success_rate": float(np.mean(scenario1["success"]))
        },
        "altered": {
            "avg_reward": float(np.mean(scenario2["rewards"])),
            "avg_deliveries": float(np.mean(scenario2["deliveries"])),
            "success_rate": float(np.mean(scenario2["success"]))
        },
        "noisy": {
            "avg_reward": float(np.mean(scenario3["rewards"])),
            "avg_deliveries": float(np.mean(scenario3["deliveries"])),
            "success_rate": float(np.mean(scenario3["success"]))
        }
    }
    
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('DQN Test Evaluation - Quantitative Metrics', fontsize=16, fontweight='bold')
    
    scenarios = ["Unseen\nEpisodes", "Altered\nEnvironment", "Noisy\nObservations"]
    rewards = [np.mean(scenario1["rewards"]), np.mean(scenario2["rewards"]), np.mean(scenario3["rewards"])]
    deliveries = [np.mean(scenario1["deliveries"]), np.mean(scenario2["deliveries"]), np.mean(scenario3["deliveries"])]
    success_rates = [np.mean(scenario1["success"])*100, np.mean(scenario2["success"])*100, np.mean(scenario3["success"])*100]
    
    axes[0,0].bar(scenarios, rewards, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0,0].set_ylabel('Avg Reward', fontsize=11, fontweight='bold')
    axes[0,0].set_title('Average Episodic Reward', fontsize=12, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    axes[0,1].bar(scenarios, deliveries, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0,1].set_ylabel('Avg Deliveries', fontsize=11, fontweight='bold')
    axes[0,1].set_ylim([0, 5.5])
    axes[0,1].set_title('Average Deliveries (out of 5)', fontsize=12, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    axes[0,2].bar(scenarios, success_rates, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0,2].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[0,2].set_ylim([0, 110])
    axes[0,2].set_title('Success Rate (5/5 Deliveries)', fontsize=12, fontweight='bold')
    axes[0,2].grid(True, alpha=0.3, axis='y')
    
    axes[1,0].boxplot([scenario1["rewards"], scenario2["rewards"], scenario3["rewards"]], labels=scenarios)
    axes[1,0].set_ylabel('Reward Distribution', fontsize=11, fontweight='bold')
    axes[1,0].set_title('Reward Stability & Variance', fontsize=12, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    axes[1,1].boxplot([scenario1["deliveries"], scenario2["deliveries"], scenario3["deliveries"]], labels=scenarios)
    axes[1,1].set_ylabel('Deliveries Distribution', fontsize=11, fontweight='bold')
    axes[1,1].set_ylim([0, 5.5])
    axes[1,1].set_title('Delivery Consistency', fontsize=12, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    axes[1,2].axis('off')
    summary_text = f"""
TEST EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━
Unseen: {rewards[0]:.1f}  {success_rates[0]:.0f}%
Altered: {rewards[1]:.1f}  {success_rates[1]:.0f}%
Noisy: {rewards[2]:.1f}  {success_rates[2]:.0f}%
    """
    axes[1,2].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                  family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / "test_evaluation_plots.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Results saved to: {results_dir}/test_results.json")
    print(f"✓ Plots saved to: {results_dir}/test_evaluation_plots.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    step4_test_evaluation()
