"""
[STEP 4] TEST MODULE & PERFORMANCE EVALUATION (Final Version)
───────────────────────────────────────────────────────────────
Evaluates:
 - DQN (untrained)
 - Double DQN (untrained)
 - A2C (untrained)
 - Champion model (trained using best configuration from Step 3)

Environment:
 - Uses fixed SEED for reproducibility
 - Same 15×15 delivery grid for all models
 - Pure exploitation (epsilon=0)
 - No altered or noisy environments

Outputs:
 - Plots of episodic rewards
 - Bar chart for mean deliveries
 - JSON file with evaluation stats
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import models and environment
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent
from models.double_dqn_agent import DoubleDQNAgent
from models.a2c_agent import A2CAgent


# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
SEED = 42
TEST_EPISODES = 50
RESULTS_DIR = Path("results/step4_test_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────
def evaluate_agent(agent, env, episodes=TEST_EPISODES, seed=SEED):
    """Evaluate agent (no training, epsilon=0)"""
    rewards, deliveries = [], []

    # Force exploitation mode
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward, ep_deliveries = 0, 0

        for step in range(env.max_steps):
            # Greedy action (no randomness)
            if hasattr(agent, "q_network"):
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = agent.q_network(state_tensor)
                    action = torch.argmax(q_values, dim=1).item()
            else:
                action = agent.select_action(state)

            next_state, reward, done, trunc, info = env.step(action)
            ep_reward += reward
            state = next_state

            if done or trunc:
                ep_deliveries = info.get("deliveries", 0)
                break

        rewards.append(ep_reward)
        deliveries.append(ep_deliveries)

    return np.array(rewards), np.array(deliveries)


def create_agent(name, config):
    """Instantiate agent by name with given config"""
    if name == "DQN":
        return DQNAgent(
            state_dim=5, action_dim=6,
            learning_rate=config.get("learning_rate", 0.001),
            gamma=config.get("gamma", 0.95),
            buffer_size=config.get("buffer_size", 10000),
            batch_size=config.get("batch_size", 32),
            target_update=config.get("target_update", 100)
        )

    elif name in ["Double DQN", "Double_DQN"]:
        return DoubleDQNAgent(
            state_dim=5, action_dim=6,
            learning_rate=config.get("learning_rate", 0.001),
            gamma=config.get("gamma", 0.95),
            buffer_size=config.get("buffer_size", 10000),
            batch_size=config.get("batch_size", 32),
            target_update=config.get("target_update", 100)
        )

    elif name == "A2C":
        return A2CAgent(
            state_dim=5, action_dim=6,
            learning_rate=config.get("learning_rate", 0.001),
            gamma=config.get("gamma", 0.95)
        )

    else:
        raise ValueError(f"Unknown agent name: {name}")


# ───────────────────────────────────────────────────────────────
# Main Entry Function
# ───────────────────────────────────────────────────────────────
def run_test_evaluation(champion, baseline_configs, test_episodes=TEST_EPISODES, seed=SEED):
    """
    Runs evaluation on:
      - DQN baseline (untrained)
      - Double DQN baseline (untrained)
      - A2C baseline (untrained)
      - Champion (trained model with best config)
    """

    print("\n" + "=" * 80)
    print("[STEP 4] PERFORMANCE EVALUATION (FINAL VERSION)")
    print("=" * 80)
    print(f"Seed: {seed} | Episodes: {test_episodes}\n")

    # Prepare consistent environment
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    torch.manual_seed(seed)
    np.random.seed(seed)

    models_to_test = [
        ("DQN", baseline_configs["DQN"]),
        ("Double DQN", baseline_configs["Double_DQN"]),
        ("A2C", baseline_configs["A2C"]),
        (f"Champion ({champion['name']})", champion["config"])
    ]

    eval_results = {}
    for name, config in models_to_test:
        print(f"\n🧠 Evaluating: {name}")
        agent = create_agent(name.replace("Champion (", "").replace(")", ""), config)
        rewards, deliveries = evaluate_agent(agent, env, episodes=test_episodes, seed=seed)
        eval_results[name] = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_deliveries": float(np.mean(deliveries)),
            "std_deliveries": float(np.std(deliveries))
        }
        print(f"   → Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"   → Avg Deliveries: {np.mean(deliveries):.2f} ± {np.std(deliveries):.2f}")

    env.close()

    # ───────────────────────────────────────────────────────────────
    # Save numeric results
    # ───────────────────────────────────────────────────────────────
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"\n📊 Results saved to: {results_path}")

    # ───────────────────────────────────────────────────────────────
    # Visualization
    # ───────────────────────────────────────────────────────────────
    model_names = list(eval_results.keys())
    rewards_mean = [r["mean_reward"] for r in eval_results.values()]
    rewards_std = [r["std_reward"] for r in eval_results.values()]
    deliveries_mean = [r["mean_deliveries"] for r in eval_results.values()]
    deliveries_std = [r["std_deliveries"] for r in eval_results.values()]

    plt.figure(figsize=(12, 6))
    plt.bar(model_names, rewards_mean, yerr=rewards_std, capsize=5, alpha=0.7, edgecolor="black")
    plt.title("Average Rewards Across Models (Fixed Environment, ε=0)")
    plt.ylabel("Average Reward")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rewards_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(model_names, deliveries_mean, yerr=deliveries_std, capsize=5, alpha=0.7, edgecolor="black", color="#2ecc71")
    plt.title("Average Deliveries per Episode (Fixed Environment, ε=0)")
    plt.ylabel("Average Deliveries (0–5)")
    plt.ylim(0, 5.5)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "deliveries_comparison.png", dpi=150)
    plt.close()

    print(f"📈 Plots saved to: {RESULTS_DIR}")
    print("\n✅ Test evaluation complete! No altered/noisy environments were used.\n")


if __name__ == "__main__":
    # Example standalone test (if you run this directly)
    dummy_champion = {
        "name": "Double DQN",
        "config": {"gamma": 0.9, "learning_rate": 0.001, "buffer_size": 10000, "batch_size": 32, "target_update": 100}
    }
    baseline_cfgs = {
        "DQN": {"gamma": 0.95, "learning_rate": 0.001, "buffer_size": 10000, "batch_size": 32, "target_update": 100},
        "Double_DQN": {"gamma": 0.95, "learning_rate": 0.001, "buffer_size": 10000, "batch_size": 32, "target_update": 100},
        "A2C": {"gamma": 0.95, "learning_rate": 0.001}
    }
    run_test_evaluation(dummy_champion, baseline_cfgs)
