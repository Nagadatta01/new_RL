import numpy as np
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.double_dqn_agent import DoubleDQNAgent

def train_champion_double_dqn():
    runs = 1
    env_config = {
        'grid_size': 15,
        'num_restaurants': 3,
        'num_customers': 5
    }
    hyperparams = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'buffer_size': 10000,
        'batch_size': 32,
        'target_update_freq': 100,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.995
    }
    for run in range(runs):
        print(f"Run {run+1}")
        env = RealisticDeliveryEnvironment(**env_config)
        agent = DoubleDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            epsilon_start=hyperparams['epsilon_start'],
            epsilon_end=hyperparams['epsilon_end'],
            epsilon_decay=hyperparams['epsilon_decay'],
            buffer_size=hyperparams['buffer_size'],
            batch_size=hyperparams['batch_size'],
            target_update=hyperparams['target_update_freq']
        )
        rewards = []
        deliveries = []
        for ep in range(100):
            state, _ = env.reset(seed=42 + run*1000 + ep)
            ep_reward = 0
            ep_deliveries = 0
            for step in range(200):
                action = agent.select_action(state)
                next_state, reward, done, trunc, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done or trunc)
                agent.train_step()
                state = next_state
                ep_reward += reward
                if done or trunc:
                    ep_deliveries = info.get('deliveries', 0)
                    break
            rewards.append(ep_reward)
            deliveries.append(ep_deliveries)
            if (ep + 1) % 25 == 0:
                avg_r = np.mean(rewards[-25:])
                avg_d = np.mean(deliveries[-25:])
                print(f"Run {run+1} | Ep {ep+1:3d} | Reward: {avg_r:.2f} | Deliveries: {avg_d:.2f}/5")
        env.close()

train_champion_double_dqn()
