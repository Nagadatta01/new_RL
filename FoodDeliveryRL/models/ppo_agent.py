import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.0003):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.action_dim = action_dim
        
        self.memory = []
        self.gamma = 0.99
        self.epsilon_clip = 0.2
        self.epochs = 4
    
    def select_action(self, state: np.ndarray, training: bool = True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.network(state)
            probs = torch.softmax(logits, dim=-1)
        
        if training:
            action = torch.multinomial(probs, 1).item()
        else:
            action = probs.argmax().item()
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.memory) < 32:
            return 0.0
        
        # Sample batch
        batch = self.memory[-32:]
        states = torch.FloatTensor([s for s, _, _, _, _ in batch]).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        
        # Compute advantages
        _, values = self.network(states)
        advantages = rewards.unsqueeze(1) - values.detach()
        
        # PPO update
        for _ in range(self.epochs):
            logits, values = self.network(states)
            probs = torch.softmax(logits, dim=-1)
            action_probs = probs.gather(1, actions.unsqueeze(1))
            
            actor_loss = -(torch.log(action_probs) * advantages).mean()
            critic_loss = advantages.pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        torch.save(self.network.state_dict(), path)
    
    def load_model(self, path: str):
        self.network.load_state_dict(torch.load(path))
