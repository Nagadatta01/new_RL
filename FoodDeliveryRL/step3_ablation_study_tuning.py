"""
[STEP 3B] ABLATION STUDY - HYPERPARAMETER TUNING (After Baseline)
Tune one parameter at a time
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

PARAM_RANGES = {
    "learning_rate": [0.0001, 0.001, 0.01],
    "gamma": [0.90, 0.95, 0.99],
    "buffer_size": [1000, 10000, 50000],
    "batch_size": [16, 32, 64],
    "target_update_freq": [50, 100, 500]
}


def train_single_run(config, run_id, num_episodes=100):
    """Train DQN - single run"""
    torch.manual_seed(SEED + run_id)
    np.random.seed(SEED + run_id)
    
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        **config,
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
                env.render()
            
            if done:
                break
        
        rewards.append(episode_reward)
        deliveries.append(env.deliveries)
        losses.append(agent.training_losses[-1] if agent.training_losses else 0)
        epsilons.append(agent.epsilon)
        
        if (episode + 1) % 25 == 0:
            print(f"      Run {run_id+1} | Ep {episode+1:3d} | Reward: {np.mean(rewards[-25:]):7.2f}")
    
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


def step3_ablation_study():
    """Step 3B: Ablation Study - Tune ONE parameter at a time"""
    
    print("\n" + "="*80)
    print("[STEP 3B] ABLATION STUDY - HYPERPARAMETER TUNING")
    print(f"Visualization: {'ENABLED ✅' if VISUALIZE else 'DISABLED ⚡ (FAST)'}")
    print("="*80)
    
    # Load baseline for reference
    baseline_file = Path("results/step3_baseline/baseline_results.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        print(f"\n✓ Baseline Mean Reward: {baseline_data['aggregate_results']['mean_reward']:.2f}")
    
    results_dir = Path("results/step3_ablation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    num_runs = 3
    all_results = {}
    best_config = BASELINE_CONFIG.copy()
    
    # ===== ABLATION 1: Learning Rate =====
    print(f"\n{'='*70}")
    print("ABLATION 1: LEARNING RATE (keep others FIXED)")
    print(f"{'='*70}")
    
    lr_results = {}
    for lr in PARAM_RANGES["learning_rate"]:
        config = BASELINE_CONFIG.copy()
        config["learning_rate"] = lr
        print(f"\n  Testing LR={lr} (3 independent runs)...")
        
        runs = [train_single_run(config, i, 100) for i in range(num_runs)]
        lr_results[lr] = {
            "avg_reward": np.mean([r["final_reward"] for r in runs]),
            "std_reward": np.std([r["final_reward"] for r in runs]),
            "runs": runs
        }
        print(f"  ✓ Result: {lr_results[lr]['avg_reward']:.2f}±{lr_results[lr]['std_reward']:.2f}")
    
    best_lr = max(lr_results, key=lambda x: lr_results[x]["avg_reward"])
    best_config["learning_rate"] = best_lr
    all_results["learning_rate"] = lr_results
    print(f"\n  ✓ BEST Learning Rate: {best_lr}")
    
    # ===== ABLATION 2: Gamma =====
    print(f"\n{'='*70}")
    print("ABLATION 2: DISCOUNT FACTOR (keep LR FIXED to best)")
    print(f"{'='*70}")
    
    gamma_results = {}
    for gamma in PARAM_RANGES["gamma"]:
        config = BASELINE_CONFIG.copy()
        config["learning_rate"] = best_lr
        config["gamma"] = gamma
        print(f"\n  Testing γ={gamma} (3 independent runs)...")
        
        runs = [train_single_run(config, i, 100) for i in range(num_runs)]
        gamma_results[gamma] = {
            "avg_reward": np.mean([r["final_reward"] for r in runs]),
            "std_reward": np.std([r["final_reward"] for r in runs]),
            "runs": runs
        }
        print(f"  ✓ Result: {gamma_results[gamma]['avg_reward']:.2f}±{gamma_results[gamma]['std_reward']:.2f}")
    
    best_gamma = max(gamma_results, key=lambda x: gamma_results[x]["avg_reward"])
    best_config["gamma"] = best_gamma
    all_results["gamma"] = gamma_results
    print(f"\n  ✓ BEST Discount Factor: {best_gamma}")
    
    # ===== ABLATION 3: Buffer Size =====
    print(f"\n{'='*70}")
    print("ABLATION 3: BUFFER SIZE (keep LR, γ FIXED to best)")
    print(f"{'='*70}")
    
    buffer_results = {}
    for buffer_size in PARAM_RANGES["buffer_size"]:
        config = BASELINE_CONFIG.copy()
        config["learning_rate"] = best_lr
        config["gamma"] = best_gamma
        config["buffer_size"] = buffer_size
        print(f"\n  Testing Buffer={buffer_size} (3 independent runs)...")
        
        runs = [train_single_run(config, i, 100) for i in range(num_runs)]
        buffer_results[buffer_size] = {
            "avg_reward": np.mean([r["final_reward"] for r in runs]),
            "std_reward": np.std([r["final_reward"] for r in runs]),
            "runs": runs
        }
        print(f"  ✓ Result: {buffer_results[buffer_size]['avg_reward']:.2f}±{buffer_results[buffer_size]['std_reward']:.2f}")
    
    best_buffer = max(buffer_results, key=lambda x: buffer_results[x]["avg_reward"])
    best_config["buffer_size"] = best_buffer
    all_results["buffer_size"] = buffer_results
    print(f"\n  ✓ BEST Buffer Size: {best_buffer}")
    
    # ===== ABLATION 4: Batch Size =====
    print(f"\n{'='*70}")
    print("ABLATION 4: BATCH SIZE (keep previous best FIXED)")
    print(f"{'='*70}")
    
    batch_results = {}
    for batch_size in PARAM_RANGES["batch_size"]:
        config = BASELINE_CONFIG.copy()
        config["learning_rate"] = best_lr
        config["gamma"] = best_gamma
        config["buffer_size"] = best_buffer
        config["batch_size"] = batch_size
        print(f"\n  Testing Batch={batch_size} (3 independent runs)...")
        
        runs = [train_single_run(config, i, 100) for i in range(num_runs)]
        batch_results[batch_size] = {
            "avg_reward": np.mean([r["final_reward"] for r in runs]),
            "std_reward": np.std([r["final_reward"] for r in runs]),
            "runs": runs
        }
        print(f"  ✓ Result: {batch_results[batch_size]['avg_reward']:.2f}±{batch_results[batch_size]['std_reward']:.2f}")
    
    best_batch = max(batch_results, key=lambda x: batch_results[x]["avg_reward"])
    best_config["batch_size"] = best_batch
    all_results["batch_size"] = batch_results
    print(f"\n  ✓ BEST Batch Size: {best_batch}")
    
    # ===== ABLATION 5: Target Update =====
    print(f"\n{'='*70}")
    print("ABLATION 5: TARGET UPDATE (keep all best FIXED)")
    print(f"{'='*70}")
    
    update_results = {}
    for update_freq in PARAM_RANGES["target_update_freq"]:
        config = BASELINE_CONFIG.copy()
        config["learning_rate"] = best_lr
        config["gamma"] = best_gamma
        config["buffer_size"] = best_buffer
        config["batch_size"] = best_batch
        config["target_update_freq"] = update_freq
        print(f"\n  Testing Update={update_freq} (3 independent runs)...")
        
        runs = [train_single_run(config, i, 100) for i in range(num_runs)]
        update_results[update_freq] = {
            "avg_reward": np.mean([r["final_reward"] for r in runs]),
            "std_reward": np.std([r["final_reward"] for r in runs]),
            "runs": runs
        }
        print(f"  ✓ Result: {update_results[update_freq]['avg_reward']:.2f}±{update_results[update_freq]['std_reward']:.2f}")
    
    best_update = max(update_results, key=lambda x: update_results[x]["avg_reward"])
    best_config["target_update_freq"] = best_update
    all_results["target_update_freq"] = update_results
    print(f"\n  ✓ BEST Target Update: {best_update}")
    
    # ===== SAVE RESULTS =====
    print(f"\n{'='*70}")
    print("FINAL TUNED CONFIGURATION")
    print(f"{'='*70}")
    
    print(f"\n✓ Best Hyperparameters Found:")
    for param, value in best_config.items():
        old_value = BASELINE_CONFIG[param]
        changed = " (CHANGED)" if old_value != value else " (same)"
        print(f"  • {param}: {value}{changed}")
    
    json_results = {}
    for param in all_results:
        json_results[param] = {
            str(k): {
                "avg_reward": float(v["avg_reward"]),
                "std_reward": float(v["std_reward"])
            }
            for k, v in all_results[param].items()
        }
    
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump({
            "ablation_results": json_results,
            "baseline_config": BASELINE_CONFIG,
            "best_config": best_config,
            "visualization_enabled": VISUALIZE
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_dir}/ablation_results.json")
    print("\n" + "="*80 + "\n")
    
    return all_results, best_config


if __name__ == "__main__":
    results, best = step3_ablation_study()
