import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

class FoodDeliveryGridEnvironment(gym.Env):
    """
    Multi-agent Food Delivery Grid World Environment
    - Grid size: 10x10
    - Agent locations: Agent 1 and Agent 2 start at different positions
    - Restaurants: Randomly placed
    - Customers: Randomly placed
    - Orders: To be delivered from restaurant to customer
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, grid_size: int = 10, 
                 num_restaurants: int = 3, 
                 num_customers: int = 5,
                 render_mode: str = None):
        """
        Initialize environment
        
        Args:
            grid_size: Size of square grid (grid_size x grid_size)
            num_restaurants: Number of restaurants
            num_customers: Number of customers
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.num_restaurants = num_restaurants
        self.num_customers = num_customers
        self.render_mode = render_mode
        
        # Grid world state
        self.agent1_pos = np.array([0, 0])
        self.agent2_pos = np.array([grid_size - 1, grid_size - 1])
        
        self.restaurant_positions = []
        self.customer_positions = []
        self.orders = {}  # {order_id: {"restaurant": pos, "customer": pos}}
        self.delivered = {}
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay, 5=pickup, 6=deliver
        self.action_space = spaces.Discrete(7)
        
        # State space: [agent1_x, agent1_y, agent2_x, agent2_y, 
        #              restaurant_grid, customer_grid, order_grid]
        # Flattened to: 4 + grid_size*grid_size*3 = 4 + 300 = 304 for 10x10 grid
        self.state_dim = 4 + (grid_size * grid_size * 3)
        self.observation_space = spaces.Box(
            low=0, 
            high=max(grid_size, 1), 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.max_steps = 500
        self.episode_reward = 0
        
    def _generate_environment(self):
        """Generate random restaurant, customer, and order locations"""
        # Reset positions
        self.restaurant_positions = []
        self.customer_positions = []
        self.orders = {}
        self.delivered = {}
        
        # Generate restaurants
        for i in range(self.num_restaurants):
            pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
            self.restaurant_positions.append(pos)
        
        # Generate customers
        for i in range(self.num_customers):
            pos = np.array([
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            ])
            self.customer_positions.append(pos)
        
        # Generate orders (each customer has one order from random restaurant)
        for customer_id in range(self.num_customers):
            restaurant_id = np.random.randint(0, self.num_restaurants)
            self.orders[customer_id] = {
                "restaurant": restaurant_id,
                "customer": customer_id,
                "delivered": False
            }
    
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        state = []
        
        # Agent positions (4 values)
        state.extend(self.agent1_pos)
        state.extend(self.agent2_pos)
        
        # Restaurant grid (grid_size x grid_size)
        restaurant_grid = np.zeros((self.grid_size, self.grid_size))
        for i, rest_pos in enumerate(self.restaurant_positions):
            restaurant_grid[rest_pos[0], rest_pos[1]] = 1
        state.extend(restaurant_grid.flatten().tolist())
        
        # Customer grid (grid_size x grid_size)
        customer_grid = np.zeros((self.grid_size, self.grid_size))
        for i, cust_pos in enumerate(self.customer_positions):
            customer_grid[cust_pos[0], cust_pos[1]] = 1
        state.extend(customer_grid.flatten().tolist())
        
        # Order status grid (grid_size x grid_size)
        order_grid = np.zeros((self.grid_size, self.grid_size))
        for order_id, order_info in self.orders.items():
            if not order_info["delivered"]:
                cust_pos = self.customer_positions[order_info["customer"]]
                order_grid[cust_pos[0], cust_pos[1]] = 1
        state.extend(order_grid.flatten().tolist())
        
        return np.array(state, dtype=np.float32)
    
    def _is_valid_pos(self, pos: np.ndarray) -> bool:
        """Check if position is within grid bounds"""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def _move_agent(self, agent_pos: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """
        Move agent based on action
        Returns: new_position, reward
        """
        new_pos = agent_pos.copy()
        movement_reward = -0.1  # Small penalty for movement
        
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Down
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == 2:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # Right
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 4:  # Stay
            movement_reward = -0.05
        
        return new_pos, movement_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in environment
        
        Args:
            action: Action for agent (0-6)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {}
        
        # Parse action for both agents (simplified: alternate or combine)
        agent_action = action % 7
        
        # Move Agent 1
        if agent_action < 5:
            self.agent1_pos, move_reward = self._move_agent(
                self.agent1_pos, 
                agent_action
            )
            reward += move_reward
        
        # Check if Agent 1 is at a restaurant (pickup)
        if agent_action == 5:
            for rest_id, rest_pos in enumerate(self.restaurant_positions):
                if np.array_equal(self.agent1_pos, rest_pos):
                    reward += 1.0  # Reward for reaching restaurant
                    info["pickup"] = True
        
        # Check if Agent 1 is at a customer (delivery)
        if agent_action == 6:
            for order_id, order_info in list(self.orders.items()):
                if not order_info["delivered"]:
                    cust_pos = self.customer_positions[order_info["customer"]]
                    if np.array_equal(self.agent1_pos, cust_pos):
                        self.orders[order_id]["delivered"] = True
                        reward += 10.0  # High reward for successful delivery
                        info["delivery"] = True
        
        # Check terminal condition: all orders delivered
        all_delivered = all(order["delivered"] for order in self.orders.values())
        if all_delivered:
            terminated = True
            reward += 5.0  # Bonus for completing all deliveries
        
        self.episode_reward += reward
        observation = self._get_state()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.agent1_pos = np.array([0, 0], dtype=np.int32)
        self.agent2_pos = np.array([
            self.grid_size - 1, 
            self.grid_size - 1
        ], dtype=np.int32)
        
        self._generate_environment()
        self.step_count = 0
        self.episode_reward = 0
        
        observation = self._get_state()
        info = {}
        
        return observation, info
    
    def render(self):
        """Render environment (placeholder)"""
        if self.render_mode == "human":
            print(f"Agent1: {self.agent1_pos}, Agent2: {self.agent2_pos}")
            print(f"Restaurants: {self.restaurant_positions}")
            print(f"Customers: {self.customer_positions}")
