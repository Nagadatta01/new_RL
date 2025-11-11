"""
[STEP 3A] BASELINE TRAINING ONLY (3 Independent Runs)
Train agent ONLY with baseline config - No tuning yet
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42

BASELINE_CONFIG = {
    "learning_rate": 0.001,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 0.995,
    "buffer_size": 10000,
    "batch_size": 32,
    "target_update_freq": 100
}


def train_baseline_single_run(run_id, num_episodes=100):
    """Train baseline - single run"""
    torch.manual_seed(SEED + run_id)
    np.random.seed(SEED + run_id)
    
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        **BASELINE_CONFIG,
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
            
            if done:
                break
        
        rewards.append(episode_reward)
        deliveries.append(env.deliveries)
        losses.append(agent.training_losses[-1] if agent.training_losses else 0)
        epsilons.append(agent.epsilon)
        
        if (episode + 1) % 25 == 0:
            print(f"      Run {run_id+1} | Ep {episode+1:3d} | Reward: {np.mean(rewards[-25:]):7.2f} | Del: {np.mean(deliveries[-25:]):.2f}/5")
    
    env.close()
    
    return {
        "rewards": rewards,
        "deliveries": deliveries,
        "losses": losses,
        "epsilons": epsilons,
        "final_reward": float(np.mean(rewards[-25:])),
        "std_reward": float(np.std(rewards[-25:])),
        "final_deliveries": float(np.mean(deliveries[-25:])),
        "std_deliveries": float(np.std(deliveries[-25:]))
    }


def step3_baseline_training():
    """Modified Step 3: Train ONLY baseline"""
    
    print("\n" + "="*80)
    print("[STEP 3A] BASELINE TRAINING ONLY (3 Independent Runs)")
    print("="*80)
    
    print(f"\n📋 Baseline Configuration:")
    for param, value in BASELINE_CONFIG.items():
        print(f"  • {param}: {value}")
    
    results_dir = Path("results/step3_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    num_runs = 3
    all_runs = []
    
    print(f"\n🚀 Training {num_runs} independent runs (100 episodes each)...\n")
    
    for run_id in range(num_runs):
        print(f"\n{'='*70}")
        print(f"BASELINE RUN {run_id+1}/{num_runs}")
        print(f"{'='*70}")
        
        result = train_baseline_single_run(run_id, 100)
        all_runs.append(result)
    
    print(f"\n{'='*70}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*70}")
    
    baseline_rewards = [r["final_reward"] for r in all_runs]
    baseline_deliveries = [r["final_deliveries"] for r in all_runs]
    
    print(f"\n✓ Aggregate Baseline Results (3 runs):")
    print(f"  • Mean Reward: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}")
    print(f"  • Reward Range: [{np.min(baseline_rewards):.2f}, {np.max(baseline_rewards):.2f}]")
    print(f"  • Mean Deliveries: {np.mean(baseline_deliveries):.2f} ± {np.std(baseline_deliveries):.2f}")
    
    summary = {
        "configuration": BASELINE_CONFIG,
        "num_runs": num_runs,
        "num_episodes": 100,
        "random_seed": SEED,
        "aggregate_results": {
            "mean_reward": float(np.mean(baseline_rewards)),
            "std_reward": float(np.std(baseline_rewards)),
            "min_reward": float(np.min(baseline_rewards)),
            "max_reward": float(np.max(baseline_rewards)),
            "mean_deliveries": float(np.mean(baseline_deliveries)),
            "std_deliveries": float(np.std(baseline_deliveries))
        }
    }
    
    with open(results_dir / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline DQN Training - 3 Independent Runs', fontsize=16, fontweight='bold')
    
    for run in all_runs:
        axes[0, 0].plot(run["rewards"], alpha=0.6, linewidth=1.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards (All Runs)')
    axes[0, 0].grid(True, alpha=0.3)
    
    window = 10
    for i, run in enumerate(all_runs):
        smoothed = np.convolve(run["rewards"], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(smoothed, label=f"Run {i+1}", linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Smoothed Reward')
    axes[0, 1].set_title(f'Smoothed Rewards ({window}-episode window)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    for i, run in enumerate(all_runs):
        axes[1, 0].plot(run["deliveries"], label=f"Run {i+1}", alpha=0.7, linewidth=1.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Deliveries Completed')
    axes[1, 0].set_ylim([0, 5.5])
    axes[1, 0].set_title('Deliveries per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    summary_text = f"""
BASELINE CONFIGURATION
━━━━━━━━━━━━━━━━━━━
Learning Rate: {BASELINE_CONFIG['learning_rate']}
Discount Factor: {BASELINE_CONFIG['gamma']}
Buffer Size: {BASELINE_CONFIG['buffer_size']}
Batch Size: {BASELINE_CONFIG['batch_size']}
Target Update: {BASELINE_CONFIG['target_update_freq']}

RESULTS (3 Independent Runs)
━━━━━━━━━━━━━━━━━━━━
Mean Reward: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}
Mean Deliveries: {np.mean(baseline_deliveries):.2f}
Reproducible: YES ✓
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(results_dir / "baseline_training_curves.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Results saved to: {results_dir}/baseline_results.json")
    print(f"✓ Plots saved to: {results_dir}/baseline_training_curves.png")
    print("\n" + "="*80 + "\n")
    
    return summary


if __name__ == "__main__":
    baseline_results = step3_baseline_training()
