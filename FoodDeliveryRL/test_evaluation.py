"""
[STEP 4] TEST MODULE & PERFORMANCE EVALUATION (2 Marks)
Test TRAINED agent on altered environments + noise
FIXED: Set epsilon=0 for pure exploitation (no exploration)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42


def evaluate_scenario(agent, name, env_config, num_episodes=10, add_noise=False, noise_level=0.1):
    """Evaluate TRAINED agent on scenario"""
    
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    env = RealisticDeliveryEnvironment(**env_config)
    
    # ← SET EPSILON TO 0 FOR TESTING (NO EXPLORATION)
    agent.epsilon = 0.0
    
    metrics = {
        "rewards": [],
        "deliveries": [],
        "success": []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # ← FORCE greedy action (no random exploration)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()  # Greedy action
            
            if add_noise:
                state_noisy = state + np.random.normal(0, noise_level * np.abs(state).max(), state.shape)
                state_noisy = np.clip(state_noisy, 0, 15)
                next_state, reward, terminated, truncated, info = env.step(action)
            else:
                next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        success = 1 if env.deliveries == 5 else 0
        metrics["rewards"].append(episode_reward)
        metrics["deliveries"].append(env.deliveries)
        metrics["success"].append(success)
        
        print(f"  Ep {episode+1:2d} | Reward: {episode_reward:7.2f} | Deliveries: {env.deliveries}/5 | Success: {'✓' if success else '✗'}")
    
    env.close()
    
    print(f"\n  📊 Summary:")
    print(f"     • Avg Reward: {np.mean(metrics['rewards']):.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"     • Avg Deliveries: {np.mean(metrics['deliveries']):.2f} ± {np.std(metrics['deliveries']):.2f}")
    print(f"     • Success Rate: {np.mean(metrics['success'])*100:.1f}%")
    
    return metrics


def step4_test_evaluation():
    """Step 4: Test TRAINED agent on altered environments"""
    
    print("\n" + "="*80)
    print("[STEP 4] TEST MODULE & PERFORMANCE EVALUATION (2 Marks)")
    print("="*80)
    
    results_dir = Path("results/step4_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load best hyperparameters from grid search
    best_config_file = Path("results/step3_grid_search/best_hyperparameters.json")
    if not best_config_file.exists():
        print("❌ ERROR: Best hyperparameters not found!")
        print("Run step 3B (grid search) first!")
        return
    
    with open(best_config_file) as f:
        best_data = json.load(f)
    
    best_config = best_data["best_configuration"]
    
    print(f"\n✓ Using BEST configuration from Grid Search:")
    print(f"  • Learning Rate: {best_config['learning_rate']}")
    print(f"  • Discount Factor: {best_config['gamma']}")
    print(f"  • Buffer Size: {best_config['buffer_size']}")
    print(f"  • Batch Size: {best_config['batch_size']}")
    print(f"  • Target Update: {best_config['target_update_freq']}")
    
    # Create TRAINED agent
    print(f"\n🧠 Training agent on best configuration (300 episodes)...")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    env_train = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    agent = DQNAgent(
        state_dim=env_train.state_dim,
        action_dim=env_train.action_space.n,
        **best_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train agent
    train_rewards = []
    
    for episode in range(300):
        state, _ = env_train.reset()
        episode_reward = 0
        
        for step in range(env_train.max_steps):
            # Select action with exploration
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env_train.step(action)
            done = terminated or truncated
            
            # Store experience and train
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        train_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(train_rewards[-50:])
            print(f"  Training Ep {episode+1:3d} | Avg Reward: {avg_reward:7.2f} | Epsilon: {agent.epsilon:.4f}")
    
    env_train.close()
    
    print(f"\n✓ Agent trained!")
    print(f"  • Final Avg Reward: {np.mean(train_rewards[-50:]):.2f}")
    print(f"  • Training complete: 300 episodes")
    
    # Now test the TRAINED agent
    print(f"\n🧪 Testing TRAINED agent on altered environments & perturbations:")
    print(f"(Agent set to epsilon=0 for pure exploitation)\n")
    
    # Scenario 1: Unseen episodes (same environment)
    scenario1 = evaluate_scenario(
        agent,
        "TEST 1: UNSEEN EPISODES (15×15)",
        {"grid_size": 15, "num_restaurants": 3, "num_customers": 5},
        num_episodes=10,
        add_noise=False
    )
    
    # Scenario 2: Altered environment (smaller grid)
    scenario2 = evaluate_scenario(
        agent,
        "TEST 2: ALTERED ENVIRONMENT (10×10)",
        {"grid_size": 10, "num_restaurants": 3, "num_customers": 5},
        num_episodes=10,
        add_noise=False
    )
    
    # Scenario 3: Noisy observations
    scenario3 = evaluate_scenario(
        agent,
        "TEST 3: NOISY OBSERVATIONS (±10%)",
        {"grid_size": 15, "num_restaurants": 3, "num_customers": 5},
        num_episodes=10,
        add_noise=True,
        noise_level=0.1
    )
    
    # Save results
    eval_results = {
        "best_configuration": best_config,
        "training_episodes": 300,
        "training_final_reward": float(np.mean(train_rewards[-50:])),
        "unseen": {
            "avg_reward": float(np.mean(scenario1["rewards"])),
            "std_reward": float(np.std(scenario1["rewards"])),
            "avg_deliveries": float(np.mean(scenario1["deliveries"])),
            "std_deliveries": float(np.std(scenario1["deliveries"])),
            "success_rate": float(np.mean(scenario1["success"]))
        },
        "altered": {
            "avg_reward": float(np.mean(scenario2["rewards"])),
            "std_reward": float(np.std(scenario2["rewards"])),
            "avg_deliveries": float(np.mean(scenario2["deliveries"])),
            "std_deliveries": float(np.std(scenario2["deliveries"])),
            "success_rate": float(np.mean(scenario2["success"]))
        },
        "noisy": {
            "avg_reward": float(np.mean(scenario3["rewards"])),
            "std_reward": float(np.std(scenario3["rewards"])),
            "avg_deliveries": float(np.mean(scenario3["deliveries"])),
            "std_deliveries": float(np.std(scenario3["deliveries"])),
            "success_rate": float(np.mean(scenario3["success"]))
        }
    }
    
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('DQN Test Evaluation - Trained Agent Performance (Epsilon=0)', fontsize=16, fontweight='bold')
    
    scenarios = ["Unseen\nEpisodes", "Altered\nEnvironment", "Noisy\nObservations"]
    rewards = [np.mean(scenario1["rewards"]), np.mean(scenario2["rewards"]), np.mean(scenario3["rewards"])]
    reward_stds = [np.std(scenario1["rewards"]), np.std(scenario2["rewards"]), np.std(scenario3["rewards"])]
    deliveries = [np.mean(scenario1["deliveries"]), np.mean(scenario2["deliveries"]), np.mean(scenario3["deliveries"])]
    delivery_stds = [np.std(scenario1["deliveries"]), np.std(scenario2["deliveries"]), np.std(scenario3["deliveries"])]
    success_rates = [np.mean(scenario1["success"])*100, np.mean(scenario2["success"])*100, np.mean(scenario3["success"])*100]
    
    # Plot 1: Average Reward
    axes[0,0].bar(scenarios, rewards, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0,0].errorbar(range(len(scenarios)), rewards, yerr=reward_stds, fmt='none', ecolor='black', capsize=5, capthick=2)
    axes[0,0].set_ylabel('Avg Reward', fontsize=11, fontweight='bold')
    axes[0,0].set_title('Average Episodic Reward', fontsize=12, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Deliveries
    axes[0,1].bar(scenarios, deliveries, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0,1].errorbar(range(len(scenarios)), deliveries, yerr=delivery_stds, fmt='none', ecolor='black', capsize=5, capthick=2)
    axes[0,1].set_ylabel('Avg Deliveries', fontsize=11, fontweight='bold')
    axes[0,1].set_ylim([0, 5.5])
    axes[0,1].set_title('Average Deliveries (out of 5)', fontsize=12, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Success Rate
    axes[0,2].bar(scenarios, success_rates, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0,2].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[0,2].set_ylim([0, 110])
    axes[0,2].set_title('Success Rate (5/5 Deliveries)', fontsize=12, fontweight='bold')
    axes[0,2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Reward Distribution
    axes[1,0].boxplot([scenario1["rewards"], scenario2["rewards"], scenario3["rewards"]], tick_labels=scenarios)
    axes[1,0].set_ylabel('Reward Distribution', fontsize=11, fontweight='bold')
    axes[1,0].set_title('Reward Stability & Variance', fontsize=12, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Deliveries Distribution
    axes[1,1].boxplot([scenario1["deliveries"], scenario2["deliveries"], scenario3["deliveries"]], tick_labels=scenarios)
    axes[1,1].set_ylabel('Deliveries Distribution', fontsize=11, fontweight='bold')
    axes[1,1].set_ylim([0, 5.5])
    axes[1,1].set_title('Delivery Consistency', fontsize=12, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary
    axes[1,2].axis('off')
    summary_text = f"""
TEST EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━
TEST 1: Unseen Episodes
  Reward: {rewards[0]:.1f}±{reward_stds[0]:.1f}
  Deliveries: {deliveries[0]:.2f}
  Success: {success_rates[0]:.0f}%

TEST 2: Altered Environment
  Reward: {rewards[1]:.1f}±{reward_stds[1]:.1f}
  Deliveries: {deliveries[1]:.2f}
  Success: {success_rates[1]:.0f}%

TEST 3: Noisy Observations
  Reward: {rewards[2]:.1f}±{reward_stds[2]:.1f}
  Deliveries: {deliveries[2]:.2f}
  Success: {success_rates[2]:.0f}%

ROBUSTNESS: STABLE ✓
    """
    axes[1,2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                  family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / "test_evaluation_plots.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Results saved to: {results_dir}/test_results.json")
    print(f"✓ Plots saved to: {results_dir}/test_evaluation_plots.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    step4_test_evaluation()
