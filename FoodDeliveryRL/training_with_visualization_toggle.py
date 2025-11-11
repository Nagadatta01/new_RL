"""
[3] TRAINING & TUNING PROTOCOL WITH VISUALIZATION TOGGLE
- Grid search over 3 hyperparameters
- 3 independent runs per configuration
- VISUALIZATION: Toggle button ON/OFF
- Training curves logged for each run
- Plots saved automatically
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import os
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42

# ================================
# ← VISUALIZATION TOGGLE BUTTON
# ================================
VISUALIZE = os.getenv('VISUALIZE', '0') == '1'
RENDER_EVERY = 30 if VISUALIZE else 0
# ================================


def train_single_run(config, run_id, scenario_id, num_episodes=100):
    """Train DQN with specific configuration on specific scenario"""
    torch.manual_seed(SEED + run_id + scenario_id * 100)
    np.random.seed(SEED + run_id + scenario_id * 100)
    
    # Setup environment based on scenario
    if scenario_id == 1:
        env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
        scenario_name = "Normal"
    elif scenario_id == 2:
        env = RealisticDeliveryEnvironment(grid_size=10, num_restaurants=3, num_customers=5)
        scenario_name = "Restricted"
    else:
        env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
        scenario_name = "Noisy"
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=config["epsilon_decay"],
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    rewards = []
    deliveries = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # ADD NOISE IF SCENARIO 3
            if scenario_id == 3:
                next_state_noisy = next_state + np.random.normal(0, 0.1 * np.abs(next_state).max(), next_state.shape)
                next_state_noisy = np.clip(next_state_noisy, 0, 15)
            else:
                next_state_noisy = next_state
            
            agent.store_experience(state, action, reward, next_state_noisy, done)
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            # VISUALIZATION WITH TOGGLE
            if VISUALIZE and (episode + 1) % RENDER_EVERY == 0:
                env.render()
            
            if done:
                break
        
        rewards.append(episode_reward)
        deliveries.append(env.deliveries)
        losses.append(agent.training_losses[-1] if agent.training_losses else 0)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print(f"    Run {run_id+1} | Ep {episode+1:3d} | Reward: {avg_reward:7.2f} | Del: {np.mean(deliveries[-20:]):.2f}/5")
    
    env.close()
    
    return {
        "rewards": rewards,
        "deliveries": deliveries,
        "losses": losses,
        "final_reward": np.mean(rewards[-20:]),
        "final_deliveries": np.mean(deliveries[-20:])
    }


def hyperparameter_tuning():
    """Systematic tuning of 3 hyperparameters across 3 scenarios"""
    
    print("\n" + "="*80)
    print("[3] TRAINING & TUNING PROTOCOL")
    print(f"Visualization: {'ENABLED ✅' if VISUALIZE else 'DISABLED ⚡ (FAST)'}")
    print("="*80)
    
    # ===== 3 HYPERPARAMETERS TO TUNE =====
    learning_rates = [0.0001, 0.001, 0.01]
    gammas = [0.90, 0.95, 0.99]
    epsilon_decays = [0.99, 0.995, 0.999]
    
    # ===== 3 SCENARIOS =====
    scenarios = {
        1: "Normal (15×15)",
        2: "Restricted (10×10)",
        3: "Noisy (±10%)"
    }
    
    num_runs = 3
    num_episodes = 100
    
    print(f"\n🔍 TUNING CONFIGURATION:")
    print(f"  • Learning Rates: {learning_rates}")
    print(f"  • Gamma Values: {gammas}")
    print(f"  • Epsilon Decay: {epsilon_decays}")
    print(f"  • Scenarios: {list(scenarios.values())}")
    print(f"  • Independent Runs per Config: {num_runs}")
    print(f"  • Episodes per Run: {num_episodes}")
    print(f"  • Total Experiments: {len(learning_rates) * len(gammas) * len(epsilon_decays) * 3 * num_runs} 🚀")
    
    all_results = {}
    
    # ===== GRID SEARCH =====
    for lr in learning_rates:
        for gamma in gammas:
            for epsilon_decay in epsilon_decays:
                config_key = f"LR={lr}_Gamma={gamma}_Eps={epsilon_decay}"
                config = {
                    "learning_rate": lr,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay
                }
                
                all_results[config_key] = {}
                
                print(f"\n{'='*70}")
                print(f"Config: {config_key}")
                print(f"{'='*70}")
                
                # Test on all 3 scenarios
                for scenario_id, scenario_name in scenarios.items():
                    print(f"\n  Testing on Scenario {scenario_id}: {scenario_name}")
                    
                    scenario_results = []
                    for run_id in range(num_runs):
                        print(f"    Independent Run {run_id+1}/{num_runs}...")
                        result = train_single_run(config, run_id, scenario_id, num_episodes)
                        scenario_results.append(result)
                    
                    # Aggregate scenario results
                    all_results[config_key][scenario_name] = {
                        "runs": scenario_results,
                        "avg_final_reward": np.mean([r["final_reward"] for r in scenario_results]),
                        "std_final_reward": np.std([r["final_reward"] for r in scenario_results]),
                        "avg_final_deliveries": np.mean([r["final_deliveries"] for r in scenario_results]),
                        "std_final_deliveries": np.std([r["final_deliveries"] for r in scenario_results])
                    }
                    
                    print(f"    Results: Reward={all_results[config_key][scenario_name]['avg_final_reward']:.2f}±{all_results[config_key][scenario_name]['std_final_reward']:.2f} | "
                          f"Deliveries={all_results[config_key][scenario_name]['avg_final_deliveries']:.2f}±{all_results[config_key][scenario_name]['std_final_deliveries']:.2f}")
    
    print(f"\n{'='*80}")
    print("TUNING COMPLETE - Generating Plots and Analysis")
    print(f"{'='*80}")
    
    # ===== SAVE RESULTS =====
    results_dir = Path("results/tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for config_key, scenario_data in all_results.items():
        json_results[config_key] = {}
        for scenario_name, metrics in scenario_data.items():
            json_results[config_key][scenario_name] = {
                "avg_final_reward": float(metrics["avg_final_reward"]),
                "std_final_reward": float(metrics["std_final_reward"]),
                "avg_final_deliveries": float(metrics["avg_final_deliveries"]),
                "std_final_deliveries": float(metrics["std_final_deliveries"])
            }
    
    with open(results_dir / "tuning_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    # ===== CREATE PLOTS =====
    print("\n📊 Creating comparison plots...")
    
    # Plot 1: Reward comparison across scenarios
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for scenario_idx, scenario_name in enumerate(scenarios.values()):
        rewards_data = []
        labels = []
        
        for config_key in all_results.keys():
            if scenario_name in all_results[config_key]:
                rewards_data.append(all_results[config_key][scenario_name]["avg_final_reward"])
                labels.append(config_key.split("_"))
        
        axes[scenario_idx].barh(range(len(rewards_data)), rewards_data, color='skyblue', edgecolor='navy')
        axes[scenario_idx].set_yticks(range(len(rewards_data)))
        axes[scenario_idx].set_yticklabels([f"Config {i+1}" for i in range(len(rewards_data))], fontsize=8)
        axes[scenario_idx].set_xlabel("Average Final Reward")
        axes[scenario_idx].set_title(f"Scenario: {scenario_name}")
        axes[scenario_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "comparison_rewards_all_scenarios.png", dpi=150)
    print("  ✓ Saved: comparison_rewards_all_scenarios.png")
    
    # Plot 2: Deliveries comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for scenario_idx, scenario_name in enumerate(scenarios.values()):
        deliveries_data = []
        
        for config_key in all_results.keys():
            if scenario_name in all_results[config_key]:
                deliveries_data.append(all_results[config_key][scenario_name]["avg_final_deliveries"])
        
        axes[scenario_idx].bar(range(len(deliveries_data)), deliveries_data, color='lightgreen', edgecolor='darkgreen')
        axes[scenario_idx].set_xlabel("Configuration")
        axes[scenario_idx].set_ylabel("Avg Deliveries")
        axes[scenario_idx].set_title(f"Scenario: {scenario_name}")
        axes[scenario_idx].set_ylim([0, 5])
        axes[scenario_idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / "comparison_deliveries_all_scenarios.png", dpi=150)
    print("  ✓ Saved: comparison_deliveries_all_scenarios.png")
    
    # ===== FIND BEST CONFIGURATION FOR EACH SCENARIO =====
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS FOR EACH SCENARIO")
    print(f"{'='*80}")
    
    best_configs = {}
    for scenario_name in scenarios.values():
        best_reward = -np.inf
        best_config = None
        
        for config_key, scenario_data in all_results.items():
            if scenario_name in scenario_data:
                reward = scenario_data[scenario_name]["avg_final_reward"]
                if reward > best_reward:
                    best_reward = reward
                    best_config = config_key
        
        best_configs[scenario_name] = {
            "config": best_config,
            "reward": best_reward,
            "data": all_results[best_config][scenario_name]
        }
        
        print(f"\n{scenario_name}:")
        print(f"  Best Config: {best_config}")
        print(f"  Final Reward: {best_reward:.2f}±{best_configs[scenario_name]['data']['std_final_reward']:.2f}")
        print(f"  Deliveries: {best_configs[scenario_name]['data']['avg_final_deliveries']:.2f}±{best_configs[scenario_name]['data']['std_final_deliveries']:.2f}")
    
    # Save best configs
    with open(results_dir / "best_configurations.json", "w") as f:
        json.dump({k: {
            "config": v["config"],
            "reward": float(v["reward"]),
            "deliveries": float(v["data"]["avg_final_deliveries"])
        } for k, v in best_configs.items()}, f, indent=2)
    
    print(f"\n✓ All results saved to: {results_dir}/")
    print(f"\n{'='*80}\n")
    
    return all_results, best_configs


if __name__ == "__main__":
    results, best = hyperparameter_tuning()
