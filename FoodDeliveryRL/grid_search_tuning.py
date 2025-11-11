"""
[STEP 3B] GRID SEARCH - HYPERPARAMETER TUNING (8 BEST CONFIGS)
Only 8 configurations - 30-40 minutes training
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

# ===== VISUALIZATION TOGGLE =====
VISUALIZE = os.getenv('VISUALIZE', '0') == '1'
RENDER_EVERY = 30 if VISUALIZE else 0
# ================================

BASELINE_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.95,
    "buffer_size": 10000,
    "batch_size": 32,
    "target_update_freq": 100
}

# ===== BEST 8 CONFIGURATIONS TO TEST =====
# Baseline + 7 smart variations
CONFIGS_TO_TEST = [
    {
        "config_name": "Config_1_Baseline",
        "learning_rate": 0.001,
        "gamma": 0.95,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 100
    },
    {
        "config_name": "Config_2_HighLR",
        "learning_rate": 0.01,          # ↑ Faster learning
        "gamma": 0.95,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 100
    },
    {
        "config_name": "Config_3_LowGamma",
        "learning_rate": 0.001,
        "gamma": 0.90,                  # ↓ More short-term focus
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 100
    },
    {
        "config_name": "Config_4_LargeBuffer",
        "learning_rate": 0.001,
        "gamma": 0.95,
        "buffer_size": 20000,           # ↑ More experience diversity
        "batch_size": 32,
        "target_update_freq": 100
    },
    {
        "config_name": "Config_5_LargeBatch",
        "learning_rate": 0.001,
        "gamma": 0.95,
        "buffer_size": 10000,
        "batch_size": 64,               # ↑ Better gradient estimates
        "target_update_freq": 100
    },
    {
        "config_name": "Config_6_FreqUpdate",
        "learning_rate": 0.001,
        "gamma": 0.95,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 50        # ↑ More frequent updates
    },
    {
        "config_name": "Config_7_Combo_Fast",
        "learning_rate": 0.01,          # Fast learning
        "gamma": 0.90,                  # Short-term focus
        "buffer_size": 20000,           # Large buffer
        "batch_size": 32,
        "target_update_freq": 100
    },
    {
        "config_name": "Config_8_Combo_Stable",
        "learning_rate": 0.001,
        "gamma": 0.99,                  # Long-term focus
        "buffer_size": 20000,           # Large buffer
        "batch_size": 64,               # Large batch
        "target_update_freq": 50        # Frequent updates
    }
]
# ========================================


def train_single_run(config, run_id, num_episodes=100):
    """Train DQN - single run with logging"""
    torch.manual_seed(SEED + run_id)
    np.random.seed(SEED + run_id)
    
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    rewards, deliveries, losses, epsilons = [], [], [], []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if VISUALIZE and (episode + 1) % RENDER_EVERY == 0:
                try:
                    env.render()
                except:
                    pass
            
            if done:
                break
        
        rewards.append(episode_reward)
        deliveries.append(env.deliveries)
        losses.append(agent.training_losses[-1] if agent.training_losses else 0)
        epsilons.append(agent.epsilon)
        
        if (episode + 1) % 25 == 0:
            print(f"        Run {run_id+1} | Ep {episode+1:3d} | Reward: {np.mean(rewards[-25:]):7.2f} | Del: {np.mean(deliveries[-25:]):.2f}/5")
    
    env.close()
    
    return {
        "rewards": rewards,
        "deliveries": deliveries,
        "losses": losses,
        "epsilons": epsilons,
        "final_reward": float(np.mean(rewards[-25:])),
        "std_reward": float(np.std(rewards[-25:])),
        "final_deliveries": float(np.mean(deliveries[-25:])),
        "std_deliveries": float(np.std(deliveries[-25:])),
        "max_deliveries": float(max(deliveries))
    }


def step3_grid_search():
    """Step 3B: Grid Search - Test BEST 8 configurations"""
    
    print("\n" + "="*80)
    print("[STEP 3B] GRID SEARCH - HYPERPARAMETER TUNING (BEST 8 CONFIGS)")
    print(f"Visualization: {'ENABLED ✅' if VISUALIZE else 'DISABLED ⚡ (FAST)'}")
    print("="*80)
    
    # Load baseline for reference
    baseline_file = Path("results/step3_baseline/baseline_results.json")
    baseline_reward = None
    baseline_deliveries = None
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        baseline_reward = baseline_data['aggregate_results']['mean_reward']
        baseline_deliveries = baseline_data['aggregate_results']['mean_deliveries']
        print(f"\n✓ Baseline Results:")
        print(f"  • Mean Reward: {baseline_reward:.2f}")
        print(f"  • Mean Deliveries: {baseline_deliveries:.2f}")
    
    results_dir = Path("results/step3_grid_search")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    num_runs = 3
    all_configurations = []
    
    total_configs = len(CONFIGS_TO_TEST)
    print(f"\n📊 Testing Best 8 Configurations...")
    print(f"✓ Total configurations: {total_configs}")
    print(f"✓ Total runs: {total_configs * num_runs} (3 per config)")
    print(f"✓ Total episodes: {total_configs * num_runs * 100} (800 configs × 3 runs × 100 eps)")
    print(f"✓ Estimated time: ~30-40 minutes (without visualization)\n")
    
    for config_idx, config_template in enumerate(CONFIGS_TO_TEST, 1):
        config_name = config_template.pop("config_name")
        
        print(f"\n{'='*70}")
        print(f"Configuration {config_idx}/{total_configs}: {config_name}")
        print(f"LR={config_template['learning_rate']} | γ={config_template['gamma']} | Buffer={config_template['buffer_size']} | Batch={config_template['batch_size']} | Update={config_template['target_update_freq']}")
        print(f"{'='*70}")
        
        runs = []
        for run_id in range(num_runs):
            result = train_single_run(config_template, run_id, 100)
            runs.append(result)
        
        # Aggregate results
        avg_reward = np.mean([r["final_reward"] for r in runs])
        std_reward = np.std([r["final_reward"] for r in runs])
        avg_deliveries = np.mean([r["final_deliveries"] for r in runs])
        std_deliveries = np.std([r["final_deliveries"] for r in runs])
        max_reward = np.max([r["final_reward"] for r in runs])
        
        print(f"\n  📊 Configuration Results:")
        print(f"     • Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"     • Avg Deliveries: {avg_deliveries:.2f} ± {std_deliveries:.2f}")
        print(f"     • Max Deliveries: {np.max([r['max_deliveries'] for r in runs]):.2f}")
        
        if baseline_reward:
            improvement = ((avg_reward - baseline_reward) / baseline_reward) * 100
            print(f"     • Improvement vs Baseline: {improvement:+.2f}%")
        
        config_result = {
            "config_num": config_idx,
            "config_name": config_name,
            "configuration": config_template,
            "num_runs": num_runs,
            "avg_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "avg_deliveries": float(avg_deliveries),
            "std_deliveries": float(std_deliveries),
            "max_reward": float(max_reward),
            "runs": runs
        }
        
        all_configurations.append(config_result)
    
    # Find best configuration
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS - FINDING BEST CONFIGURATION")
    print(f"{'='*80}")
    
    best_idx = np.argmax([c["avg_reward"] for c in all_configurations])
    best_result = all_configurations[best_idx]
    best_config = best_result["configuration"]
    
    print(f"\n✓ BEST Configuration: {best_result['config_name']} (Config #{best_result['config_num']})")
    print(f"  • Learning Rate: {best_config['learning_rate']}")
    print(f"  • Discount Factor (γ): {best_config['gamma']}")
    print(f"  • Buffer Size: {best_config['buffer_size']}")
    print(f"  • Batch Size: {best_config['batch_size']}")
    print(f"  • Target Update: {best_config['target_update_freq']}")
    print(f"\n✓ BEST Performance:")
    print(f"  • Avg Reward: {best_result['avg_reward']:.2f} ± {best_result['std_reward']:.2f}")
    print(f"  • Avg Deliveries: {best_result['avg_deliveries']:.2f} ± {best_result['std_deliveries']:.2f}")
    
    if baseline_reward:
        improvement = ((best_result['avg_reward'] - baseline_reward) / baseline_reward) * 100
        print(f"  • Improvement vs Baseline: {improvement:+.2f}%")
    
    # Save results
    json_data = {
        "total_configurations": total_configs,
        "num_runs_per_config": num_runs,
        "baseline_config": BASELINE_CONFIG,
        "best_configuration": best_config,
        "best_config_name": best_result['config_name'],
        "best_results": {
            "avg_reward": float(best_result['avg_reward']),
            "std_reward": float(best_result['std_reward']),
            "avg_deliveries": float(best_result['avg_deliveries']),
            "std_deliveries": float(best_result['std_deliveries']),
            "config_number": int(best_result['config_num'])
        },
        "all_configurations": [
            {
                "config_num": c["config_num"],
                "config_name": c["config_name"],
                "configuration": c["configuration"],
                "avg_reward": float(c["avg_reward"]),
                "std_reward": float(c["std_reward"]),
                "avg_deliveries": float(c["avg_deliveries"]),
                "std_deliveries": float(c["std_deliveries"])
            }
            for c in all_configurations
        ]
    }
    
    with open(results_dir / "grid_search_results.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    with open(results_dir / "best_hyperparameters.json", "w") as f:
        json.dump({
            "best_configuration_name": best_result['config_name'],
            "best_configuration": best_config,
            "performance": {
                "avg_reward": float(best_result['avg_reward']),
                "avg_deliveries": float(best_result['avg_deliveries']),
                "std_reward": float(best_result['std_reward']),
                "std_deliveries": float(best_result['std_deliveries'])
            }
        }, f, indent=2)
    
    # ===== CREATE PLOTS =====
    print(f"\n📊 Generating plots...")
    
    config_names = [c["config_name"] for c in all_configurations]
    rewards = [c["avg_reward"] for c in all_configurations]
    deliveries = [c["avg_deliveries"] for c in all_configurations]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Grid Search Results - Best 8 Configurations', fontsize=16, fontweight='bold')
    
    # Plot 1: Rewards
    colors = ['green' if r == max(rewards) else 'blue' for r in rewards]
    axes[0, 0].bar(config_names, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    if baseline_reward:
        axes[0, 0].axhline(y=baseline_reward, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_reward:.0f}')
    axes[0, 0].set_ylabel('Average Reward', fontweight='bold')
    axes[0, 0].set_title('Reward Comparison', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend()
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Deliveries
    colors = ['green' if d == max(deliveries) else 'blue' for d in deliveries]
    axes[0, 1].bar(config_names, deliveries, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    if baseline_deliveries:
        axes[0, 1].axhline(y=baseline_deliveries, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_deliveries:.1f}')
    axes[0, 1].set_ylabel('Average Deliveries', fontweight='bold')
    axes[0, 1].set_ylim([0, 5.5])
    axes[0, 1].set_title('Deliveries Comparison', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend()
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Scatter
    scatter = axes[1, 0].scatter(deliveries, rewards, c=range(len(rewards)), cmap='viridis', s=200, edgecolors='black', linewidth=2)
    axes[1, 0].scatter([best_result['avg_deliveries']], [best_result['avg_reward']], 
                      color='red', s=400, marker='*', edgecolors='black', linewidth=2, zorder=5, label='Best')
    axes[1, 0].set_xlabel('Average Deliveries', fontweight='bold')
    axes[1, 0].set_ylabel('Average Reward', fontweight='bold')
    axes[1, 0].set_title('Reward vs Deliveries', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    plt.colorbar(scatter, ax=axes[1, 0], label='Config #')
    
    # Plot 4: Summary
    axes[1, 1].axis('off')
    summary_text = f"""
GRID SEARCH SUMMARY
━━━━━━━━━━━━━━━━━━━━━
Configurations: {total_configs}
Runs per Config: {num_runs}
Total Episodes: {total_configs * num_runs * 100}
Est. Time: 30-40 min

BEST: {best_result['config_name']}
Reward: {best_result['avg_reward']:.2f}±{best_result['std_reward']:.2f}
Deliveries: {best_result['avg_deliveries']:.2f}
Improvement: {((best_result['avg_reward'] - baseline_reward) / baseline_reward * 100) if baseline_reward else 0:+.1f}%

Config Details:
LR={best_config['learning_rate']}
γ={best_config['gamma']}
Buffer={best_config['buffer_size']}
Batch={best_config['batch_size']}
Update={best_config['target_update_freq']}
    """
    
    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=9.5, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(results_dir / "grid_search_plots.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Plots saved: {results_dir}/grid_search_plots.png")
    print(f"✓ Results saved: {results_dir}/grid_search_results.json")
    
    print("\n" + "="*80 + "\n")
    
    return all_configurations, best_config, best_result


if __name__ == "__main__":
    configs, best, best_result = step3_grid_search()
