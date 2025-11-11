import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent
from models.double_dqn_agent import DoubleDQNAgent
from models.a2c_agent import A2CAgent

SEED = 42

def test_agent(agent, env, num_episodes=100):
    """Runs episodes with given agent and environment, returns reward and delivery stats."""
    rewards = []
    deliveries = []

    agent.epsilon = 0.0  # pure exploitation

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + ep)
        ep_reward = 0

        for _ in range(env.max_steps):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                if hasattr(agent, 'q_network'):
                    q_values = agent.q_network(state_tensor)
                    action = torch.argmax(q_values, dim=1).item()
                elif hasattr(agent, 'policy_network'):
                    action_probs = agent.policy_network(state_tensor)
                    action = torch.argmax(action_probs, dim=1).item()
                else:
                    action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(ep_reward)
        deliveries.append(env.deliveries if hasattr(env, 'deliveries') else info.get('deliveries', 0))

    return np.array(rewards), np.array(deliveries)


def step4_test_evaluation():
    print("\n" + "="*80)
    print("[STEP 4] TEST MODULE & PERFORMANCE EVALUATION (UPDATED)")
    print("="*80)

    results_dir = Path("results/step4_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)

    champion_file = Path("results/step3_champion/champion_selection.json")
    if not champion_file.exists():
        print("❌ ERROR: Champion info not found! Run Step 3 first.")
        return

    with open(champion_file) as f:
        champion_data = json.load(f)

    champion_name = champion_data['name']
    champion_config = champion_data['config']

    baseline_config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'buffer_size': 10000,
        'batch_size': 32,
        'target_update': 100
    }

    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)

    # Baseline agents untrained with baseline config
    dqn_baseline = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=baseline_config['learning_rate'],
        gamma=baseline_config['gamma'],
        buffer_size=baseline_config['buffer_size'],
        batch_size=baseline_config['batch_size'],
        target_update=baseline_config['target_update'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    double_dqn_baseline = DoubleDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=baseline_config['learning_rate'],
        gamma=baseline_config['gamma'],
        buffer_size=baseline_config['buffer_size'],
        batch_size=baseline_config['batch_size'],
        target_update=baseline_config['target_update'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    a2c_baseline = A2CAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        learning_rate=baseline_config['learning_rate'],
        gamma=baseline_config['gamma'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Champion agent
    if champion_name == 'DQN':
        champion_agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **champion_config
        )
    elif champion_name == 'Double DQN':
        champion_agent = DoubleDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **champion_config
        )
    elif champion_name == 'A2C':
        champion_agent = A2CAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **champion_config
        )
    else:
        print(f"❌ Unknown champion algorithm: {champion_name}")
        env.close()
        return

    # Train champion agent for 300 episodes
    print(f"Training champion agent {champion_name} for 300 episodes...")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_rewards = []
    for episode in range(300):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for step in range(env.max_steps):
            action = champion_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            champion_agent.store_experience(state, action, reward, next_state, done)
            champion_agent.train_step()
            episode_reward += reward
            state = next_state
            if done:
                break

        train_rewards.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(train_rewards[-50:])
            print(f"Training Ep {episode+1:3d} | Avg Reward: {avg_reward:7.2f} | Epsilon: {champion_agent.epsilon:.4f}")

    print(f"Champion agent training complete! Final avg reward (last 50 eps): {np.mean(train_rewards[-50:]):.2f}")

    # Test all agents on 100 episodes each
    print("\nTesting baseline and champion agents on 100 episodes... (epsilon=0)")

    dqn_rewards, dqn_deliveries = test_agent(dqn_baseline, env, 100)
    double_dqn_rewards, double_dqn_deliveries = test_agent(double_dqn_baseline, env, 100)
    a2c_rewards, a2c_deliveries = test_agent(a2c_baseline, env, 100)
    champion_rewards, champion_deliveries = test_agent(champion_agent, env, 100)

    env.close()

    # Plot comparison
    labels = ['DQN Baseline', 'Double DQN Baseline', 'A2C Baseline', f'{champion_name} Champion']
    avg_rewards = [np.mean(dqn_rewards), np.mean(double_dqn_rewards), np.mean(a2c_rewards), np.mean(champion_rewards)]
    std_rewards = [np.std(dqn_rewards), np.std(double_dqn_rewards), np.std(a2c_rewards), np.std(champion_rewards)]

    avg_deliveries = [np.mean(dqn_deliveries), np.mean(double_dqn_deliveries), np.mean(a2c_deliveries), np.mean(champion_deliveries)]
    std_deliveries = [np.std(dqn_deliveries), np.std(double_dqn_deliveries), np.std(a2c_deliveries), np.std(champion_deliveries)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10,6))

    bars1 = ax1.bar(x - width/2, avg_rewards, width, yerr=std_rewards, label='Avg Reward', color='#3498db', capsize=5)
    ax1.set_ylabel('Avg Reward', color='#3498db', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_ylim(0, max(avg_rewards) * 1.2)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, avg_deliveries, width, yerr=std_deliveries, label='Avg Deliveries', color='#2ecc71', capsize=5)
    ax2.set_ylabel('Avg Deliveries (out of 5)', color='#2ecc71', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.set_ylim(0, 5.5)

    plt.xticks(x, labels, rotation=15, fontsize=11)
    plt.title('Baseline vs Champion Agent Performance Comparison (100 Episodes)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.legend(handles=[bars1[0], bars2[0]], loc='upper left')

    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'baseline_vs_champion_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Comparison plot saved to: {results_dir}/baseline_vs_champion_comparison.png")

    # Save results JSON
    results = {
        'dqn_baseline': {
            'avg_reward': float(np.mean(dqn_rewards)),
            'std_reward': float(np.std(dqn_rewards)),
            'avg_deliveries': float(np.mean(dqn_deliveries)),
            'std_deliveries': float(np.std(dqn_deliveries))
        },
        'double_dqn_baseline': {
            'avg_reward': float(np.mean(double_dqn_rewards)),
            'std_reward': float(np.std(double_dqn_rewards)),
            'avg_deliveries': float(np.mean(double_dqn_deliveries)),
            'std_deliveries': float(np.std(double_dqn_deliveries))
        },
        'a2c_baseline': {
            'avg_reward': float(np.mean(a2c_rewards)),
            'std_reward': float(np.std(a2c_rewards)),
            'avg_deliveries': float(np.mean(a2c_deliveries)),
            'std_deliveries': float(np.std(a2c_deliveries))
        },
        'champion': {
            'algorithm': champion_name,
            'config': champion_config,
            'avg_reward': float(np.mean(champion_rewards)),
            'std_reward': float(np.std(champion_rewards)),
            'avg_deliveries': float(np.mean(champion_deliveries)),
            'std_deliveries': float(np.std(champion_deliveries))
        }
    }

    with open(results_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_dir}/comparison_results.json")


if __name__ == '__main__':
    step4_test_evaluation()
