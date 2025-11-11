"""
A2C (Advantage Actor-Critic) Agent
Policy-based method, good for continuous learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCriticNetwork(nn.Module):
    """Shared network with actor and critic heads"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # Shared layers
        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))
        
        # Actor output (action probabilities)
        action_probs = torch.softmax(self.actor_fc(x), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic_fc(x)
        
        return action_probs, state_value


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent
    On-policy, policy gradient method
    """
    
    def __init__(self, state_dim, action_dim,
                 learning_rate=0.001, gamma=0.9,
                 epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Episode memory (for A2C update)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def select_action(self, state):
        """Sample action from policy with epsilon-greedy exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            action_probs = action_probs.cpu().numpy()[0]
            action = np.random.choice(self.action_dim, p=action_probs)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition for episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
        # Train at episode end
        if done:
            self.train_step()
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
    
    def train_step(self):
        """A2C training step (called at episode end)"""
        if len(self.episode_states) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.episode_states).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        
        # Calculate returns (discounted rewards)
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Forward pass
        action_probs, state_values = self.network(states)
        state_values = state_values.squeeze()
        
        # Calculate advantages
        advantages = returns - state_values.detach()
        
        # Actor loss (policy gradient)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = nn.MSELoss()(state_values, returns)
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
