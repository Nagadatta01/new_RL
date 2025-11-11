import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from typing import List, Dict
import threading
import queue

class LiveTrainerVisualizer:
    """Real-time live training visualization with Gymnasium"""
    
    def __init__(self, grid_size: int = 10, max_episodes_display: int = 100):
        """
        Initialize live visualizer
        
        Args:
            grid_size: Grid size
            max_episodes_display: Max episodes to show in plot
        """
        self.grid_size = grid_size
        self.max_episodes_display = max_episodes_display
        
        # Data queues for thread-safe updates
        self.episode_data = queue.Queue()
        
        # Store data
        self.episodes_list = deque(maxlen=max_episodes_display)
        self.rewards_list = deque(maxlen=max_episodes_display)
        self.losses_list = deque(maxlen=max_episodes_display)
        self.epsilon_list = deque(maxlen=max_episodes_display)
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('DQN Training Live Dashboard (Gymnasium)', 
                         fontsize=16, fontweight='bold')
        
        # Create grid
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Environment visualization
        self.ax_env = self.fig.add_subplot(gs[0, 0])
        
        # Plot 2: Rewards
        self.ax_reward = self.fig.add_subplot(gs[0, 1])
        
        # Plot 3: Loss
        self.ax_loss = self.fig.add_subplot(gs[0, 2])
        
        # Plot 4: Epsilon
        self.ax_epsilon = self.fig.add_subplot(gs[1, 0])
        
        # Plot 5: Stats
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.axis('off')
        
        # Plot 6: Q-values distribution
        self.ax_qvalues = self.fig.add_subplot(gs[1, 2])
        
        # Setup plots
        self._setup_plots()
        
        self.env_data = None
        self.current_episode = 0
        self.current_reward = 0
        
        plt.ion()  # Interactive mode
    
    def _setup_plots(self):
        """Setup initial plot configurations"""
        
        # Rewards plot
        self.ax_reward.set_xlabel('Episode', fontsize=10)
        self.ax_reward.set_ylabel('Reward', fontsize=10)
        self.ax_reward.set_title('Episode Rewards', fontweight='bold')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Loss plot
        self.ax_loss.set_xlabel('Episode', fontsize=10)
        self.ax_loss.set_ylabel('Loss (MSE)', fontsize=10)
        self.ax_loss.set_title('Training Loss', fontweight='bold')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Epsilon plot
        self.ax_epsilon.set_xlabel('Episode', fontsize=10)
        self.ax_epsilon.set_ylabel('Epsilon (ε)', fontsize=10)
        self.ax_epsilon.set_title('Exploration Rate', fontweight='bold')
        self.ax_epsilon.grid(True, alpha=0.3)
        self.ax_epsilon.set_ylim([0, 1.1])
        
        # Environment plot
        self.ax_env.set_title('Environment State', fontweight='bold')
        self.ax_env.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_aspect('equal')
        
        # Q-values plot
        self.ax_qvalues.set_xlabel('Action', fontsize=10)
        self.ax_qvalues.set_ylabel('Q-Value', fontsize=10)
        self.ax_qvalues.set_title('Q-Values Distribution', fontweight='bold')
        self.ax_qvalues.grid(True, alpha=0.3)
    
    def update_environment(self, env_data: Dict):
        """Update environment visualization"""
        self.env_data = env_data
        self._draw_environment()
    
    def _draw_environment(self):
        """Draw environment grid with agents and targets"""
        if self.env_data is None:
            return
        
        self.ax_env.clear()
        self.ax_env.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax_env.set_aspect('equal')
        self.ax_env.set_title('Environment State', fontweight='bold')
        
        # Draw grid
        for i in range(self.grid_size + 1):
            self.ax_env.axhline(y=i - 0.5, color='gray', linestyle='-', 
                                linewidth=0.5, alpha=0.3)
            self.ax_env.axvline(x=i - 0.5, color='gray', linestyle='-', 
                                linewidth=0.5, alpha=0.3)
        
        env = self.env_data['env']
        
        # Draw restaurants
        for i, pos in enumerate(env.restaurant_positions):
            rect = patches.Rectangle((pos[1] - 0.25, pos[0] - 0.25), 0.5, 0.5,
                                     linewidth=2, edgecolor='black',
                                     facecolor='#FFD93D', zorder=3)
            self.ax_env.add_patch(rect)
            self.ax_env.text(pos[1], pos[0], f'R{i+1}', ha='center', 
                            va='center', fontsize=7, fontweight='bold')
        
        # Draw customers
        for i, pos in enumerate(env.customer_positions):
            diamond = patches.Polygon([
                [pos[1], pos[0] + 0.3],
                [pos[1] + 0.3, pos[0]],
                [pos[1], pos[0] - 0.3],
                [pos[1] - 0.3, pos[0]]
            ], closed=True, linewidth=2, edgecolor='darkgreen',
               facecolor='#6BCB77', zorder=3)
            self.ax_env.add_patch(diamond)
            self.ax_env.text(pos[1], pos[0], f'C{i+1}', ha='center',
                            va='center', fontsize=7, fontweight='bold')
        
        # Draw pending orders
        for order_id, order_info in env.orders.items():
            if not order_info["delivered"]:
                cust_pos = env.customer_positions[order_info["customer"]]
                circle = patches.Circle((cust_pos[1], cust_pos[0]), 0.5,
                                       fill=False, edgecolor='#FF6B9D',
                                       linewidth=2, linestyle='--', zorder=2)
                self.ax_env.add_patch(circle)
        
        # Draw agents
        circle1 = patches.Circle((env.agent1_pos[1], env.agent1_pos[0]), 0.3,
                                color='#FF6B6B', zorder=5)
        self.ax_env.add_patch(circle1)
        self.ax_env.text(env.agent1_pos[1], env.agent1_pos[0] - 0.6, 'Agent 1',
                        ha='center', fontsize=8, fontweight='bold')
        
        circle2 = patches.Circle((env.agent2_pos[1], env.agent2_pos[0]), 0.3,
                                color='#4ECDC4', zorder=5)
        self.ax_env.add_patch(circle2)
        self.ax_env.text(env.agent2_pos[1], env.agent2_pos[0] - 0.6, 'Agent 2',
                        ha='center', fontsize=8, fontweight='bold')
        
        # Add delivery info
        delivered = sum(1 for o in env.orders.values() if o["delivered"])
        total = len(env.orders)
        info_text = f"Deliveries: {delivered}/{total}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        self.ax_env.text(0.02, 0.98, info_text, transform=self.ax_env.transAxes,
                        fontsize=9, verticalalignment='top', bbox=props)
    
    def update_metrics(self, episode: int, reward: float, loss: float, 
                      epsilon: float, q_values: np.ndarray = None):
        """Update training metrics"""
        
        self.current_episode = episode
        self.current_reward = reward
        
        self.episodes_list.append(episode)
        self.rewards_list.append(reward)
        self.losses_list.append(loss)
        self.epsilon_list.append(epsilon)
        
        # Update reward plot
        self.ax_reward.clear()
        episodes_arr = np.array(self.episodes_list)
        rewards_arr = np.array(self.rewards_list)
        
        self.ax_reward.plot(episodes_arr, rewards_arr, 'b-', linewidth=2, label='Reward')
        
        # Add moving average
        if len(rewards_arr) >= 20:
            ma = np.convolve(rewards_arr, np.ones(20)/20, mode='valid')
            ma_episodes = episodes_arr[19:]
            self.ax_reward.plot(ma_episodes, ma, 'r--', linewidth=2, label='MA(20)')
        
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.set_title('Episode Rewards')
        self.ax_reward.grid(True, alpha=0.3)
        self.ax_reward.legend(loc='upper left', fontsize=9)
        
        # Update loss plot
        self.ax_loss.clear()
        losses_arr = np.array(self.losses_list)
        self.ax_loss.plot(episodes_arr, losses_arr, 'g-', linewidth=2)
        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Loss (MSE)')
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Update epsilon plot
        self.ax_epsilon.clear()
        epsilon_arr = np.array(self.epsilon_list)
        self.ax_epsilon.plot(episodes_arr, epsilon_arr, 'orange', linewidth=2)
        self.ax_epsilon.set_xlabel('Episode')
        self.ax_epsilon.set_ylabel('Epsilon (ε)')
        self.ax_epsilon.set_title('Exploration Rate')
        self.ax_epsilon.grid(True, alpha=0.3)
        self.ax_epsilon.set_ylim([0, 1.1])
        
        # Update statistics
        self._update_stats(reward, loss, epsilon)
        
        # Update Q-values distribution
        if q_values is not None:
            self._update_qvalues(q_values)
        
        # Redraw
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def _update_stats(self, reward: float, loss: float, epsilon: float):
        """Update statistics panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        avg_reward = np.mean(list(self.rewards_list)[-100:]) if len(self.rewards_list) >= 100 else np.mean(list(self.rewards_list))
        max_reward = np.max(list(self.rewards_list)) if self.rewards_list else 0
        
        stats_text = f"""
        LIVE TRAINING STATISTICS
        {'='*35}
        
        Episode: {self.current_episode}
        
        Current Reward: {reward:>10.2f}
        Avg Reward (100): {avg_reward:>8.2f}
        Max Reward: {max_reward:>12.2f}
        
        Loss (MSE): {loss:>11.4f}
        Epsilon (ε): {epsilon:>11.4f}
        
        Total Episodes: {len(self.episodes_list):>7}
        """
        
        self.ax_stats.text(0.1, 0.5, stats_text, fontsize=10,
                          family='monospace', verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _update_qvalues(self, q_values: np.ndarray):
        """Update Q-values distribution"""
        self.ax_qvalues.clear()
        
        actions = np.arange(len(q_values))
        self.ax_qvalues.bar(actions, q_values, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax_qvalues.set_xlabel('Action')
        self.ax_qvalues.set_ylabel('Q-Value')
        self.ax_qvalues.set_title('Q-Values Distribution')
        self.ax_qvalues.grid(True, alpha=0.3, axis='y')
    
    def close(self):
        """Close visualizer"""
        plt.close(self.fig)
