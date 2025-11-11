import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List

class DQNNetwork(nn.Module):
    """Deep Q-Network for Food Delivery RL"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 128):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, buffer_size: int = 100000):
        """
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for Food Delivery"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 device: str = "cpu"):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate (α)
            gamma: Discount factor (γ)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            device: Computing device (cpu/cuda)
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=learning_rate
        )
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Tracking
        self.training_losses = []
        self.q_values = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Track Q-values
        self.q_values.append(q_values.max().item())
        
        return q_values.argmax(1).item()
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.q_network(states_t).gather(1, actions_t)
        
        # Compute max Q(s_{t+1}, a') using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )
        
        return loss.item()
    
    def store_experience(self, 
                        state: np.ndarray, 
                        action: int, 
                        reward: float,
                        next_state: np.ndarray, 
                        done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
