import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import deque
from typing import Dict
import time


class SlowLiveVisualizer:
    """Slow-motion live visualizer - see agents move step by step"""
    
    def __init__(self, grid_size: int = 20, pause_duration: float = 0.5):
        """
        Initialize slow visualizer
        
        Args:
            grid_size: Grid size
            pause_duration: Pause between steps (seconds)
        """
        self.grid_size = grid_size
        self.pause_duration = pause_duration
        
        # Data storage
        self.episodes_list = deque(maxlen=100)
        self.rewards_list = deque(maxlen=100)
        
        # Create figure
        self.fig = plt.figure(figsize=(18, 8))
        self.fig.suptitle('🌆 Realistic Food Delivery Training (Slow Motion)', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        # Environment plot (larger)
        self.ax_env = self.fig.add_subplot(gs[0])
        
        # Rewards plot
        self.ax_reward = self.fig.add_subplot(gs[1])
        
        self._setup_plots()
        
        self.current_episode = 0
        
        plt.ion()
    
    def _setup_plots(self):
        """Setup plot configurations"""
        
        # Environment plot
        self.ax_env.set_title('🌆 City Grid Environment (Real-Time)', fontweight='bold', fontsize=14)
        self.ax_env.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_aspect('equal')
        self.ax_env.set_xlabel('X Position', fontsize=11)
        self.ax_env.set_ylabel('Y Position', fontsize=11)
        
        # Rewards plot
        self.ax_reward.set_xlabel('Episode', fontsize=10)
        self.ax_reward.set_ylabel('Reward', fontsize=10)
        self.ax_reward.set_title('📈 Training Progress', fontweight='bold')
        self.ax_reward.grid(True, alpha=0.3)
    
    def update_environment(self, env, episode: int, step: int, reward: float):
        """Update environment visualization with roads, grass, parks"""
        
        self.ax_env.clear()
        self.ax_env.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_aspect('equal')
        self.ax_env.set_title(f'🌆 Episode {episode} | Step {step} | Reward: {reward:.1f}', 
                             fontweight='bold', fontsize=14)
        self.ax_env.set_xlabel('X Position', fontsize=11)
        self.ax_env.set_ylabel('Y Position', fontsize=11)
        
        # Set background to light green (grass)
        self.ax_env.set_facecolor('#E8F5E9')
        
        # Draw grid
        for i in range(self.grid_size + 1):
            self.ax_env.axhline(y=i - 0.5, color='#CCCCCC', linestyle='-', 
                                linewidth=0.3, alpha=0.3)
            self.ax_env.axvline(x=i - 0.5, color='#CCCCCC', linestyle='-', 
                                linewidth=0.3, alpha=0.3)
        
        # Draw parks (beautiful green with pattern)
        for park_pos in env.parks:
            rect = patches.Rectangle((park_pos[1] - 0.5, park_pos[0] - 0.5), 1, 1,
                                     linewidth=0, facecolor='#4CAF50', alpha=0.6, zorder=1)
            self.ax_env.add_patch(rect)
        
        # Draw roads (gray asphalt)
        for road_pos in env.roads:
            rect = patches.Rectangle((road_pos[1] - 0.5, road_pos[0] - 0.5), 1, 1,
                                     linewidth=0, facecolor='#888888', zorder=1)
            self.ax_env.add_patch(rect)
        
        # Draw buildings (dark gray with windows)
        for building_pos in env.buildings:
            rect = patches.Rectangle((building_pos[1] - 0.5, building_pos[0] - 0.5), 1, 1,
                                     linewidth=1.5, edgecolor='#333333', 
                                     facecolor='#555555', zorder=2)
            self.ax_env.add_patch(rect)
            
            # Add window details
            self.ax_env.plot([building_pos[1] - 0.2], [building_pos[0] - 0.2], 'yo', markersize=2)
            self.ax_env.plot([building_pos[1] + 0.2], [building_pos[0] + 0.2], 'yo', markersize=2)
        
        # Draw restaurants (orange with pizza emoji style)
        for i, rest_pos in enumerate(env.restaurant_positions):
            rect = patches.Rectangle((rest_pos[1] - 0.35, rest_pos[0] - 0.35), 0.7, 0.7,
                                     linewidth=3, edgecolor='#FF6F00',
                                     facecolor='#FFB74D', zorder=4)
            self.ax_env.add_patch(rect)
            self.ax_env.text(rest_pos[1], rest_pos[0], '🍕', ha='center',
                            va='center', fontsize=12)
        
        # Draw customers (green with person emoji)
        for i, cust_pos in enumerate(env.customer_positions):
            diamond = patches.Polygon([
                [cust_pos[1], cust_pos[0] + 0.4],
                [cust_pos[1] + 0.4, cust_pos[0]],
                [cust_pos[1], cust_pos[0] - 0.4],
                [cust_pos[1] - 0.4, cust_pos[0]]
            ], closed=True, linewidth=2.5, edgecolor='#388E3C',
               facecolor='#81C784', zorder=4)
            self.ax_env.add_patch(diamond)
            self.ax_env.text(cust_pos[1], cust_pos[0], '👤', ha='center',
                            va='center', fontsize=11)
        
        # Draw pending orders (dashed circles)
        for order_id, order_info in env.orders.items():
            if not order_info["delivered"]:
                cust_pos = env.customer_positions[order_info["customer"]]
                circle = patches.Circle((cust_pos[1], cust_pos[0]), 0.6,
                                       fill=False, edgecolor='#E91E63',
                                       linewidth=2.5, linestyle='--', zorder=3)
                self.ax_env.add_patch(circle)
        
        # Draw Agent 1 (RED - Main agent)
        agent1_glow = patches.Circle((env.agent1_pos[1], env.agent1_pos[0]), 0.7,
                                    color='#FF0000', alpha=0.25, zorder=4)
        self.ax_env.add_patch(agent1_glow)
        
        agent1_circle = patches.Circle((env.agent1_pos[1], env.agent1_pos[0]), 0.4,
                                      color='#FF0000', zorder=5, linewidth=2, 
                                      edgecolor='#CC0000')
        self.ax_env.add_patch(agent1_circle)
        
        self.ax_env.text(env.agent1_pos[1], env.agent1_pos[0], '🚗', ha='center',
                        va='center', fontsize=11, fontweight='bold')
        
        self.ax_env.text(env.agent1_pos[1], env.agent1_pos[0] - 0.85, 
                        f'🚗 Agent 1\n({env.agent1_deliveries})',
                        ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.9))
        
        # Draw Agent 2 (CYAN - Secondary agent)
        agent2_glow = patches.Circle((env.agent2_pos[1], env.agent2_pos[0]), 0.7,
                                    color='#00BCD4', alpha=0.25, zorder=4)
        self.ax_env.add_patch(agent2_glow)
        
        agent2_circle = patches.Circle((env.agent2_pos[1], env.agent2_pos[0]), 0.4,
                                      color='#00BCD4', zorder=5, linewidth=2,
                                      edgecolor='#0097A7')
        self.ax_env.add_patch(agent2_circle)
        
        self.ax_env.text(env.agent2_pos[1], env.agent2_pos[0], '🚙', ha='center',
                        va='center', fontsize=11, fontweight='bold')
        
        self.ax_env.text(env.agent2_pos[1], env.agent2_pos[0] - 0.85,
                        f'🚙 Agent 2\n({env.agent2_deliveries})',
                        ha='center', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='#E0F7FA', alpha=0.9))
        
        # Info panel
        delivered = sum(1 for o in env.orders.values() if o["delivered"])
        total = len(env.orders)
        
        info_text = (
            f"📦 Deliveries: {delivered}/{total}\n"
            f"🚗 Agent 1: {env.agent1_deliveries}\n"
            f"🚙 Agent 2: {env.agent2_deliveries}\n"
            f"⏱️ Step: {step}/{env.max_steps}\n"
            f"🚧 Collisions: {env.total_collisions}"
        )
        
        props = dict(boxstyle='round', facecolor='#FFFACD', alpha=0.95, linewidth=2.5, edgecolor='#333333')
        self.ax_env.text(0.02, 0.98, info_text, transform=self.ax_env.transAxes,
                        fontsize=11, verticalalignment='top', fontweight='bold',
                        bbox=props, family='monospace')
        
        # Legend
        legend_elements = [
            patches.Patch(facecolor='#FF0000', label='🚗 Agent 1 (Trained)'),
            patches.Patch(facecolor='#00BCD4', label='🚙 Agent 2 (Random Helper)'),
            patches.Patch(facecolor='#FFB74D', label='🍕 Restaurant'),
            patches.Patch(facecolor='#81C784', label='👤 Customer'),
            patches.Patch(facecolor='#888888', label='🛣️ Road'),
            patches.Patch(facecolor='#4CAF50', label='🌳 Park'),
            patches.Patch(facecolor='#555555', label='🏢 Building'),
            patches.Patch(facecolor='white', edgecolor='#E91E63', 
                         linestyle='--', linewidth=2, label='📦 Order')
        ]
        self.ax_env.legend(handles=legend_elements, loc='lower right', 
                          fontsize=8.5, framealpha=0.95, ncol=2)
        
        # Draw
        self.fig.canvas.draw()
        plt.pause(self.pause_duration)
    
    def update_rewards(self, episode: int, reward: float):
        """Update rewards plot"""
        self.episodes_list.append(episode)
        self.rewards_list.append(reward)
        
        self.ax_reward.clear()
        episodes_arr = np.array(self.episodes_list)
        rewards_arr = np.array(self.rewards_list)
        
        self.ax_reward.plot(episodes_arr, rewards_arr, 'b-', linewidth=2)
        
        if len(rewards_arr) >= 10:
            ma = np.convolve(rewards_arr, np.ones(10)/10, mode='valid')
            ma_episodes = episodes_arr[9:]
            self.ax_reward.plot(ma_episodes, ma, 'r--', linewidth=2, label='MA(10)')
            self.ax_reward.legend()
        
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_title('📈 Training Progress')
        self.ax_reward.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
        plt.pause(0.01)
    
    def close(self):
        """Close visualizer"""
        plt.close(self.fig)
