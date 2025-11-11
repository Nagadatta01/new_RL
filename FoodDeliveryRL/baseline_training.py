"""
BASELINE TRAINING: Train DQN with default configuration
This gives you baseline results to compare against
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42

# BASELINE CONFIGURATION (from document defaults)
BASELINE_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.95,
    "buffer_size": 10000,
    "batch_size": 32,
    "target_update_freq": 100
}


def train_baseline(num_runs=3, num_episodes=100):
    """Train baseline DQN - 3 independent runs"""
    
    print("\n" + "="*80)
    print("BASELINE TRAINING - Default Configuration")
    print("="*80)
    
    print(f"\nBaseline Configuration:")
    for param, value in BASELINE_CONFIG.items():
        print(f"  • {param}: {value}")
    
    print(f"\nTraining Settings:")
    print(f"  • Number of runs: {num_runs}")
    print(f"  • Episodes per run: {num_episodes}")
    print(f"  • Random Seed: {SEED} (reproducible)")
    
    results_dir = Path("results/baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_runs = []
    
    # ===== RUN 3 INDEPENDENT BASELINE TRAININGS =====
    for run_id in range(num_runs):
        print(f"\n{'='*80}")
        print(f"BASELINE RUN {run_id+1}/{num_runs}")
        print(f"{'='*80}")
        
        torch.manual_seed(SEED + run_id)
        np.random.seed(SEED + run_id)
        
        env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
        
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            **BASELINE_CONFIG,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        rewards = []
        deliveries = []
        losses = []
        epsilons = []
        
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
                
                if done:
                    break
            
            rewards.append(episode_reward)
            deliveries.append(env.deliveries)
            losses.append(agent.training_losses[-1] if agent.training_losses else 0)
            epsilons.append(agent.epsilon)
            
            if (episode + 1) % 25 == 0:
                avg_reward = np.mean(rewards[-25:])
                avg_del = np.mean(deliveries[-25:])
                print(f"  Episode {episode+1:3d}/100 | Reward: {avg_reward:7.2f} | Deliveries: {avg_del:.2f}/5")
        
        env.close()
        
        run_result = {
            "run_id": run_id + 1,
            "seed": SEED + run_id,
            "rewards": rewards,
            "deliveries": deliveries,
            "losses": losses,
            "epsilons": epsilons,
            "final_reward": np.mean(rewards[-25:]),
            "std_reward": np.std(rewards[-25:]),
            "final_deliveries": np.mean(deliveries[-25:]),
            "std_deliveries": np.std(deliveries[-25:]),
            "max_deliveries_achieved": max(deliveries)
        }
        
        all_runs.append(run_result)
        
        print(f"\n  Run {run_id+1} Summary:")
        print(f"    Final Reward: {run_result['final_reward']:.2f} ± {run_result['std_reward']:.2f}")
        print(f"    Avg Deliveries: {run_result['final_deliveries']:.2f}/5")
        print(f"    Max Deliveries: {run_result['max_deliveries_achieved']}/5")
    
    # ===== AGGREGATE BASELINE RESULTS =====
    print(f"\n{'='*80}")
    print("BASELINE RESULTS SUMMARY (All 3 Runs)")
    print(f"{'='*80}")
    
    baseline_rewards = [r["final_reward"] for r in all_runs]
    baseline_deliveries = [r["final_deliveries"] for r in all_runs]
    
    summary = {
        "configuration": BASELINE_CONFIG,
        "num_runs": num_runs,
        "num_episodes": num_episodes,
        "random_seed": SEED,
        "per_run_results": [
            {
                "run": r["run_id"],
                "seed": r["seed"],
                "final_reward": float(r["final_reward"]),
                "std_reward": float(r["std_reward"]),
                "final_deliveries": float(r["final_deliveries"]),
                "max_deliveries": int(r["max_deliveries_achieved"])
            }
            for r in all_runs
        ],
        "aggregate_results": {
            "mean_reward": float(np.mean(baseline_rewards)),
            "std_reward": float(np.std(baseline_rewards)),
            "mean_deliveries": float(np.mean(baseline_deliveries)),
            "std_deliveries": float(np.std(baseline_deliveries)),
            "min_reward": float(np.min(baseline_rewards)),
            "max_reward": float(np.max(baseline_rewards))
        }
    }
    
    print(f"\n✓ Aggregate Baseline Results:")
    print(f"  Mean Reward: {summary['aggregate_results']['mean_reward']:.2f} ± {summary['aggregate_results']['std_reward']:.2f}")
    print(f"  Range: [{summary['aggregate_results']['min_reward']:.2f}, {summary['aggregate_results']['max_reward']:.2f}]")
    print(f"  Mean Deliveries: {summary['aggregate_results']['mean_deliveries']:.2f} ± {summary['aggregate_results']['std_deliveries']:.2f}")
    
    # ===== SAVE BASELINE RESULTS =====
    with open(results_dir / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Baseline results saved to: {results_dir}/baseline_results.json")
    
    # ===== PLOT BASELINE TRAINING CURVES =====
    print(f"\nGenerating baseline training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline DQN Training Curves (3 Independent Runs)', fontsize=16, fontweight='bold')
    
    # Plot 1: Rewards for each run
    for run in all_runs:
        axes[0, 0].plot(run["rewards"], alpha=0.6, label=f"Run {run['run_id']}", linewidth=1.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards (All Runs)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Smoothed rewards
    window = 10
    for run in all_runs:
        smoothed = np.convolve(run["rewards"], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(smoothed, label=f"Run {run['run_id']}", linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Smoothed Reward')
    axes[0, 1].set_title(f'Smoothed Rewards ({window}-episode window)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Deliveries
    for run in all_runs:
        axes[1, 0].plot(run["deliveries"], alpha=0.6, label=f"Run {run['run_id']}", linewidth=1.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Deliveries Completed')
    axes[1, 0].set_ylim([0, 5.5])
    axes[1, 0].set_title('Deliveries per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    runs = [f"Run {r['run_id']}" for r in all_runs]
    final_rewards = [r["final_reward"] for r in all_runs]
    final_deliveries = [r["final_deliveries"] for r in all_runs]
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
BASELINE CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━
Learning Rate: {BASELINE_CONFIG['learning_rate']}
Discount Factor: {BASELINE_CONFIG['gamma']}
Buffer Size: {BASELINE_CONFIG['buffer_size']}
Batch Size: {BASELINE_CONFIG['batch_size']}
Target Update: {BASELINE_CONFIG['target_update_freq']}

RESULTS (3 Independent Runs)
━━━━━━━━━━━━━━━━━━━━━
Mean Reward: {summary['aggregate_results']['mean_reward']:.2f} ± {summary['aggregate_results']['std_reward']:.2f}
Mean Deliveries: {summary['aggregate_results']['mean_deliveries']:.2f} ± {summary['aggregate_results']['std_deliveries']:.2f}
Reproducible: YES ✓
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / "baseline_training_curves.png", dpi=150)
    print(f"✓ Saved: {results_dir}/baseline_training_curves.png")
    
    print(f"\n{'='*80}\n")
    
    return summary


if __name__ == "__main__":
    baseline_results = train_baseline(num_runs=3, num_episodes=100)
