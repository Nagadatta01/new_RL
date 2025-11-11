import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class RealisticDeliveryEnvironment(gym.Env):
    """
    WORKING environment with:
    - Fixed visualization (uses your working render method)
    - Optimized rewards for better learning
    - Relaxed walkability (can walk near roads)
    - Generous pickup/delivery distances
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, grid_size: int = 15, num_restaurants: int = 3, num_customers: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.num_restaurants = num_restaurants
        self.num_customers = num_customers
        
        self.state_dim = 5
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(self.state_dim,), dtype=np.float32)
        
        self.restaurant_positions = []
        self.customer_positions = []
        self.buildings = []
        self.roads = set()
        self.adjacent_to_roads = []
        self.orders = {}
        self.carrying = None
        self.agent_pos = None
        
        self.step_count = 0
        self.max_steps = 200  # ← REDUCED from 500
        self.deliveries = 0
        self.collisions = 0
        
        self.fig = None
        self.ax = None
        
        self._setup_city()
    
    def _setup_city(self):
        """Setup simpler city"""
        self.buildings = []
        self.roads = set()
        self.adjacent_to_roads = []
        
        # ROADS
        for row in [2, 5, 8, 11]:  # ← REDUCED roads for 15x15 grid
            for col in range(self.grid_size):
                self.roads.add((row, col))
        
        for col in [2, 5, 8, 11]:
            for row in range(self.grid_size):
                self.roads.add((row, col))
        
        # FEWER BUILDINGS
        self._add_building_block(1, 1, 1, 1)
        self._add_building_block(3, 3, 1, 1)
        self._add_building_block(1, 10, 1, 1)
        self._add_building_block(10, 1, 1, 1)
        self._add_building_block(10, 10, 1, 1)
        
        # Find positions adjacent to roads
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.roads and not any(np.array_equal(np.array([x, y]), b) for b in self.buildings):
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in self.roads:
                            self.adjacent_to_roads.append(np.array([x, y]))
                            break
    
    def _add_building_block(self, start_x, start_y, width, height):
        for x in range(start_x, min(start_x + width + 1, self.grid_size)):
            for y in range(start_y, min(start_y + height + 1, self.grid_size)):
                if (x, y) not in self.roads:
                    self.buildings.append(np.array([x, y]))
    
    def _is_walkable(self, pos: np.ndarray) -> bool:
        """
        RELAXED WALKABILITY:
        - Can walk ON roads
        - Can walk NEAR roads (1 cell away)
        - Cannot walk on buildings
        """
        x, y = int(pos[0]), int(pos[1])
        
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        
        for building in self.buildings:
            if np.array_equal(np.array([x, y]), building):
                return False
        
        if (x, y) in self.roads:
            return True
        
        # Can walk near roads
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (x + dx, y + dy) in self.roads:
                    return True
        
        return False
    
    def _get_walkable_position(self):
        for _ in range(100):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if self._is_walkable(np.array([x, y])):
                return np.array([x, y])
        return np.array([self.grid_size // 2, self.grid_size // 2])
    
    def _generate_orders(self):
        self.restaurant_positions = []
        self.customer_positions = []
        self.orders = {}
        
        # Restaurants
        for _ in range(self.num_restaurants):
            pos = self._get_walkable_position()
            while any(np.array_equal(pos, r) for r in self.restaurant_positions):
                pos = self._get_walkable_position()
            self.restaurant_positions.append(pos)
        
        # Customers
        for _ in range(self.num_customers):
            pos = self._get_walkable_position()
            while (any(np.array_equal(pos, c) for c in self.customer_positions) or
                   any(np.array_equal(pos, r) for r in self.restaurant_positions)):
                pos = self._get_walkable_position()
            self.customer_positions.append(pos)
        
        # Orders
        for i in range(self.num_customers):
            self.orders[i] = {
                "restaurant": np.random.randint(0, len(self.restaurant_positions)),
                "customer": i,
                "picked_up": False,
                "delivered": False
            }
    
    def _get_state(self) -> np.ndarray:
        if self.carrying is None:
            target = None
            for order_id, order in self.orders.items():
                if not order["picked_up"]:
                    target = self.restaurant_positions[order["restaurant"]]
                    break
            if target is None:
                target = np.array([self.grid_size // 2, self.grid_size // 2])
        else:
            target = self.customer_positions[self.orders[self.carrying]["customer"]]
        
        return np.array([self.agent_pos[0], self.agent_pos[1],
                        1.0 if self.carrying is not None else 0.0,
                        target[0], target[1]], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """OPTIMIZED STEP with better rewards"""
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_steps
        old_pos = self.agent_pos.copy()
        
        # MOVEMENT
        if action < 4:
            new_pos = self.agent_pos.copy()
            if action == 0:
                new_pos[0] -= 1
            elif action == 1:
                new_pos[0] += 1
            elif action == 2:
                new_pos[1] -= 1
            elif action == 3:
                new_pos[1] += 1
            
            if not self._is_walkable(new_pos):
                reward -= 2.0
                self.collisions += 1
            else:
                self.agent_pos = new_pos
                
                # Distance reward (Manhattan)
                if self.carrying is None:
                    for order_id, order in self.orders.items():
                        if not order["picked_up"]:
                            rest_pos = self.restaurant_positions[order["restaurant"]]
                            old_dist = np.abs(old_pos - rest_pos).sum()
                            new_dist = np.abs(self.agent_pos - rest_pos).sum()
                            dist_improvement = old_dist - new_dist
                            reward += dist_improvement * 0.5
                            break
                else:
                    cust_pos = self.customer_positions[self.orders[self.carrying]["customer"]]
                    old_dist = np.abs(old_pos - cust_pos).sum()
                    new_dist = np.abs(self.agent_pos - cust_pos).sum()
                    dist_improvement = old_dist - new_dist
                    reward += dist_improvement * 0.5
        
        # PICKUP
        elif action == 4:
            picked_up = False
            for order_id, order in self.orders.items():
                if not order["picked_up"]:
                    rest_pos = self.restaurant_positions[order["restaurant"]]
                    dist = np.abs(self.agent_pos - rest_pos).sum()
                    
                    if dist <= 3.0:  # ← Generous distance
                        self.orders[order_id]["picked_up"] = True
                        self.carrying = order_id
                        reward += 20.0
                        picked_up = True
                        break
            
            if not picked_up:
                reward -= 0.5
        
        # DELIVERY
        elif action == 5:
            if self.carrying is not None:
                cust_pos = self.customer_positions[self.orders[self.carrying]["customer"]]
                dist = np.abs(self.agent_pos - cust_pos).sum()
                
                if dist <= 3.0:  # ← Generous distance
                    self.orders[self.carrying]["delivered"] = True
                    self.carrying = None
                    self.deliveries += 1
                    reward += 100.0
                else:
                    reward -= 0.5
            else:
                reward -= 0.5
        
        # COMPLETION BONUS
        if all(order["delivered"] for order in self.orders.values()):
            terminated = True
            reward += 200.0
        
        return self._get_state(), reward, terminated, truncated, {
            "deliveries": self.deliveries,
            "collisions": self.collisions
        }
    
    def render(self, mode='human'):
        """
        WORKING VISUALIZATION (using your original working code structure)
        """
        if self.fig is None:
            # ← KEY: Use plt.figure() instead of plt.subplots()
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111)
            plt.ion()
        
        self.ax.clear()
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # ← KEY: Match your working code
        self.ax.axis('off')
        
        # Background
        bg = patches.Rectangle((-1, -1), self.grid_size + 2, self.grid_size + 2,
                               facecolor='#F0F0F0', zorder=0)
        self.ax.add_patch(bg)
        
        # Roads
        for x, y in self.roads:
            rect = patches.Rectangle((y - 0.4, x - 0.4), 0.8, 0.8,
                                    facecolor='#A9A9A9', edgecolor='#696969', 
                                    linewidth=0.5, zorder=2)
            self.ax.add_patch(rect)
        
        # Road markings
        for row in [2, 5, 8, 11]:
            self.ax.plot([-0.5, self.grid_size - 0.5], [row, row], 
                        'y--', linewidth=1.5, alpha=0.5, zorder=3)
        for col in [2, 5, 8, 11]:
            self.ax.plot([col, col], [-0.5, self.grid_size - 0.5], 
                        'y--', linewidth=1.5, alpha=0.5, zorder=3)
        
        # Buildings
        for building in self.buildings:
            shadow = patches.Rectangle((building[1] - 0.43, building[0] - 0.48), 
                                      0.86, 0.86, facecolor='#505050', 
                                      alpha=0.3, zorder=3)
            self.ax.add_patch(shadow)
            
            rect = patches.Rectangle((building[1] - 0.45, building[0] - 0.45), 
                                    0.9, 0.9, facecolor='#8B4513', 
                                    edgecolor='#5C2E0A', linewidth=2,
                                    alpha=0.95, zorder=4)
            self.ax.add_patch(rect)
            
            # Windows
            for wx in [-0.15, 0.15]:
                for wy in [-0.15, 0.15]:
                    window = patches.Rectangle((building[1] + wx - 0.05, 
                                               building[0] + wy - 0.05),
                                              0.1, 0.1, facecolor='#FFD700', 
                                              alpha=0.7, zorder=5)
                    self.ax.add_patch(window)
        
        # Restaurants
        for i, rest in enumerate(self.restaurant_positions):
            circle = patches.Circle((rest[1], rest[0]), 0.35, color='#FF4500',
                                   edgecolor='#8B0000', linewidth=2.5, 
                                   alpha=0.95, zorder=6)
            self.ax.add_patch(circle)
            self.ax.text(rest[1], rest[0], '🍕', ha='center', va='center', 
                        fontsize=14, zorder=7)
            self.ax.text(rest[1], rest[0] - 0.65, f'R{i+1}', ha='center', 
                        fontsize=9, weight='bold', color='#8B0000', zorder=7)
        
        # Customers
        for i, cust in enumerate(self.customer_positions):
            if i < len(self.orders):
                order = self.orders[i]
                
                if order["delivered"]:
                    circle = patches.Circle((cust[1], cust[0]), 0.28, 
                                          color='#00AA00', edgecolor='#006600', 
                                          linewidth=2, alpha=0.9, zorder=6)
                    self.ax.add_patch(circle)
                    self.ax.text(cust[1], cust[0], '✓', ha='center', va='center',
                               fontsize=12, zorder=7, weight='bold', color='white')
                else:
                    circle = patches.Circle((cust[1], cust[0]), 0.28, 
                                          color='#FFD700', edgecolor='#B8860B', 
                                          linewidth=1.5, alpha=0.9, zorder=6)
                    self.ax.add_patch(circle)
                    self.ax.text(cust[1], cust[0], '👤', ha='center', va='center',
                               fontsize=11, zorder=7)
                
                self.ax.text(cust[1], cust[0] - 0.55, f'C{i+1}', ha='center', 
                           fontsize=8, weight='bold', color='#B8860B', zorder=7)
        
        # Agent
        agent_color = '#FF1744' if self.carrying is None else '#00E676'
        agent_circle = patches.Circle((self.agent_pos[1], self.agent_pos[0]), 0.4,
                                     color=agent_color, edgecolor='#990000', 
                                     linewidth=3, alpha=0.95, zorder=9)
        self.ax.add_patch(agent_circle)
        self.ax.text(self.agent_pos[1], self.agent_pos[0], '🚗', 
                    ha='center', va='center', fontsize=16, zorder=10, weight='bold')
        
        if self.carrying is not None:
            self.ax.text(self.agent_pos[1], self.agent_pos[0] - 0.7, '📦',
                        ha='center', fontsize=12, zorder=10)
        
        # Status bar
        status = f'🌆 DELIVERY | Step: {self.step_count}/{self.max_steps} | ' \
                f'✓ Delivered: {self.deliveries}/{self.num_customers} | ' \
                f'💥 Collisions: {self.collisions}'
        self.ax.text(self.grid_size / 2, -0.5, status, ha='center', fontsize=11,
                    weight='bold', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', edgecolor='black', linewidth=2))
        
        plt.xlim(-1.5, self.grid_size + 1.5)
        plt.draw()  # ← KEY: From your working code
        plt.pause(0.02)  # ← KEY: From your working code
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._generate_orders()
        self.agent_pos = self._get_walkable_position()
        self.carrying = None
        self.step_count = 0
        self.deliveries = 0
        self.collisions = 0
        return self._get_state(), {}
