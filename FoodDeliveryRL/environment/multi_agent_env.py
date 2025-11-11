import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple

class MultiAgentFoodDeliveryEnv(gym.Env):
    """Multi-Agent Food Delivery Environment"""
    
    def __init__(self, 
                 grid_size: int = 10,
                 num_agents: int = 2,
                 num_restaurants: int = 3,
                 num_customers: int = 5):
        
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_restaurants = num_restaurants
        self.num_customers = num_customers
        
        # Agent positions
        self.agent_positions = {
            i: np.array([
                i * (grid_size // num_agents), 
                i * (grid_size // num_agents)
            ]) 
            for i in range(num_agents)
        }
        
        # Shared state
        self.restaurant_positions = []
        self.customer_positions = []
        self.orders = {}
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(7)
        self.state_dim = 4 + (grid_size * grid_size * 3)
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size, 1),
            shape=(self.state_dim,), dtype=np.float32
        )
        
        self.step_count = 0
        self.max_steps = 500
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Re-initialize agent positions
        for i in range(self.num_agents):
            self.agent_positions[i] = np.array([
                i % self.grid_size,
                i // self.grid_size
            ])
        
        # Generate environment
        self._generate_environment()
        self.step_count = 0
        
        observations = {
            i: self._get_agent_state(i) 
            for i in range(self.num_agents)
        }
        
        return observations, {}
    
    def step(self, actions: Dict[int, int]):
        """
        Step all agents
        
        Args:
            actions: {agent_id: action}
            
        Returns:
            observations, rewards, terminated, truncated, info
        """
        self.step_count += 1
        
        rewards = {i: 0.0 for i in range(self.num_agents)}
        info = {i: {} for i in range(self.num_agents)}
        
        # Move each agent
        for agent_id, action in actions.items():
            if action < 5:
                new_pos = self._move_agent(
                    self.agent_positions[agent_id], 
                    action
                )
                self.agent_positions[agent_id] = new_pos
                rewards[agent_id] -= 0.1  # Movement cost
        
        # Check deliveries and update rewards
        for agent_id in range(self.num_agents):
            for order_id, order_info in self.orders.items():
                if not order_info["delivered"]:
                    cust_pos = self.customer_positions[order_info["customer"]]
                    if np.array_equal(self.agent_positions[agent_id], cust_pos):
                        if actions[agent_id] == 6:  # Delivery action
                            self.orders[order_id]["delivered"] = True
                            rewards[agent_id] += 10.0
                            info[agent_id]["delivery"] = True
        
        # Shared reward for team success
        all_delivered = all(o["delivered"] for o in self.orders.values())
        if all_delivered:
            for agent_id in range(self.num_agents):
                rewards[agent_id] += 5.0
        
        terminated = all_delivered
        truncated = self.step_count >= self.max_steps
        
        observations = {
            i: self._get_agent_state(i) 
            for i in range(self.num_agents)
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _get_agent_state(self, agent_id: int) -> np.ndarray:
        """Get state for specific agent"""
        state = []
        
        # Add all agent positions
        for i in range(self.num_agents):
            state.extend(self.agent_positions[i])
        
        # Pad remaining slots if fewer than 2 agents
        while len(state) < 4:
            state.extend([0, 0])
        
        # Add grids (restaurant, customer, order)
        for _ in range(3):
            state.extend(np.zeros(self.grid_size * self.grid_size).tolist())
        
        return np.array(state[:self.state_dim], dtype=np.float32)
    
    def _move_agent(self, pos: np.ndarray, action: int) -> np.ndarray:
        """Move agent"""
        new_pos = pos.copy()
        
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        
        return new_pos
    
    def _generate_environment(self):
        """Generate restaurant and customer locations"""
        self.restaurant_positions = [
            np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
            for _ in range(self.num_restaurants)
        ]
        
        self.customer_positions = [
            np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
            for _ in range(self.num_customers)
        ]
        
        self.orders = {
            i: {
                "restaurant": np.random.randint(0, self.num_restaurants),
                "customer": i,
                "delivered": False
            }
            for i in range(self.num_customers)
        }
