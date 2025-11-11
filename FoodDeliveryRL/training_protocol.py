"""
[3] TUNING & TRAINING PROTOCOL (2 Marks)
- Systematic grid search
- Multiple independent runs
- Logging and visualization
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42


def train_with_config(config, run_id, num_episodes=100):
    """Train DQN with specific configuration"""
    torch.manual_seed(SEED + run_id)
    np.random.seed(SEED + run_id)
    
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
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
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_del = np.mean(deliveries[-20:])
            print(f"  Run {run_id}: Ep {episode+1:3d} | Reward: {avg_reward:7.2f} | Del: {avg_del:.2f}/5")
    
    env.close()
    
    return {
        "rewards": rewards,
        "deliveries": deliveries,
        "losses": losses,
        "epsilons": epsilons,
        "final_reward": np.mean(rewards[-20:]),
        "final_deliveries": np.mean(deliveries[-20:])
    }


def tuning_protocol():
    """Systematic hyperparameter tuning with multiple runs"""
    
    print("\n" + "="*80)
    print("[3] TUNING & TRAINING PROTOCOL")
    print("="*80)
    
    # ===== BASELINE CONFIG =====
    baseline_config = {
        "learning_rate": 0.001,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 100
    }
    
    # ===== GRID SEARCH =====
    configs_to_test = {
        "baseline": baseline_config,
        "high_lr": {**baseline_config, "learning_rate": 0.01},
        "low_lr": {**baseline_config, "learning_rate": 0.0001},
        "high_gamma": {**baseline_config, "gamma": 0.99},
        "low_gamma": {**baseline_config, "gamma": 0.9},
    }
    
    results = {}
    num_runs = 3  # 3 independent runs per config
    num_episodes = 100
    
    print(f"\n✓ Grid Search Configuration:")
    print(f"  • Configurations: {len(configs_to_test)}")
    print(f"  • Independent Runs: {num_runs} per config")
    print(f"  • Episodes per Run: {num_episodes}")
    print(f"  • Total Runs: {len(configs_to_test) * num_runs}")
    
    for config_name, config in configs_to_test.items():
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {config_name}")
        print(f"{'='*60}")
        
        run_results = []
        for run_id in range(num_runs):
            print(f"\n  Independent Run {run_id + 1}/{num_runs}")
            result = train_with_config(config, run_id, num_episodes)
            run_results.append(result)
        
        # Aggregate results
        all_rewards = [r["rewards"] for r in run_results]
        all_deliveries = [r["deliveries"] for r in run_results]
        
        results[config_name] = {
            "config": config,
            "runs": run_results,
            "avg_final_reward": np.mean([r["final_reward"] for r in run_results]),
            "std_final_reward": np.std([r["final_reward"] for r in run_results]),
            "avg_final_deliveries": np.mean([r["final_deliveries"] for r in run_results]),
            "std_final_deliveries": np.std([r["final_deliveries"] for r in run_results])
        }
        
        print(f"\n  Results Summary:")
        print(f"    • Avg Final Reward: {results[config_name]['avg_final_reward']:.2f} ± {results[config_name]['std_final_reward']:.2f}")
        print(f"    • Avg Final Deliveries: {results[config_name]['avg_final_deliveries']:.2f} ± {results[config_name]['std_final_deliveries']:.2f}")
    
    # ===== SAVE RESULTS =====
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    summary_data = {config_name: {
        "config": results[config_name]["config"],
        "avg_final_reward": float(results[config_name]["avg_final_reward"]),
        "std_final_reward": float(results[config_name]["std_final_reward"]),
        "avg_final_deliveries": float(results[config_name]["avg_final_deliveries"]),
        "std_final_deliveries": float(results[config_name]["std_final_deliveries"])
    } for config_name in results}
    
    with open(analysis_dir / "tuning_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✓ Tuning results saved to: {analysis_dir}/tuning_results.json")
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    tuning_protocol()
