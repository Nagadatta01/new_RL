import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

class EnvironmentRenderer:
    """Real-time environment visualization"""
    
    def __init__(self, grid_size: int = 10, output_dir: str = "logs/env_render"):
        """
        Initialize environment renderer
        
        Args:
            grid_size: Size of grid
            output_dir: Directory to save render frames
        """
        self.grid_size = grid_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for elements
        self.colors = {
            'empty': 'white',
            'agent1': '#FF6B6B',      # Red
            'agent2': '#4ECDC4',      # Cyan
            'restaurant': '#FFD93D',  # Yellow
            'customer': '#6BCB77',    # Green
            'order': '#FF6B9D'        # Pink
        }
        
        self.fig = None
        self.ax = None
    
    def render_episode(self, 
                       env,
                       episode: int,
                       step: int,
                       reward: float) -> np.ndarray:
        """
        Render current environment state
        
        Args:
            env: Environment instance
            episode: Current episode number
            step: Current step in episode
            reward: Current episode reward
            
        Returns:
            Rendered image as numpy array
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.ax.clear()
        
        # Draw grid
        self._draw_grid()
        
        # Draw elements
        self._draw_restaurants(env.restaurant_positions)
        self._draw_customers(env.customer_positions)
        self._draw_agent(env.agent1_pos, label="Agent 1", color=self.colors['agent1'])
        self._draw_agent(env.agent2_pos, label="Agent 2", color=self.colors['agent2'])
        self._draw_orders(env)
        
        # Add labels and info
        self._add_info_panel(episode, step, reward, env)
        
        # Set title and labels
        self.ax.set_title(f'Grid World Visualization - Episode {episode}, Step {step}', 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Position', fontsize=12)
        self.ax.set_ylabel('Y Position', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        frame_path = self.output_dir / f"episode_{episode:04d}_step_{step:04d}.png"
        self.fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        
        return frame_path
    
    def _draw_grid(self):
        """Draw background grid"""
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            self.ax.axhline(y=i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            self.ax.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Fill background
        self.ax.set_facecolor('#F0F0F0')
    
    def _draw_agent(self, pos: np.ndarray, label: str, color: str):
        """Draw agent on grid"""
        circle = patches.Circle((pos[1], pos[0]), 0.3, color=color, zorder=5)
        self.ax.add_patch(circle)
        
        # Add label
        self.ax.text(pos[1], pos[0] - 0.6, label, ha='center', fontsize=9, fontweight='bold')
    
    def _draw_restaurants(self, restaurant_positions: List[np.ndarray]):
        """Draw restaurants on grid"""
        for i, pos in enumerate(restaurant_positions):
            # Draw square for restaurant
            rect = patches.Rectangle((pos[1] - 0.25, pos[0] - 0.25), 0.5, 0.5, 
                                     linewidth=2, edgecolor='black', 
                                     facecolor=self.colors['restaurant'], zorder=3)
            self.ax.add_patch(rect)
            
            # Add label
            self.ax.text(pos[1], pos[0], f'R{i+1}', ha='center', va='center', 
                        fontsize=8, fontweight='bold')
    
    def _draw_customers(self, customer_positions: List[np.ndarray]):
        """Draw customers on grid"""
        for i, pos in enumerate(customer_positions):
            # Draw diamond for customer
            diamond = patches.Polygon([
                [pos[1], pos[0] + 0.3],      # Top
                [pos[1] + 0.3, pos[0]],      # Right
                [pos[1], pos[0] - 0.3],      # Bottom
                [pos[1] - 0.3, pos[0]]       # Left
            ], closed=True, linewidth=2, edgecolor='darkgreen', 
               facecolor=self.colors['customer'], zorder=3)
            self.ax.add_patch(diamond)
            
            # Add label
            self.ax.text(pos[1], pos[0], f'C{i+1}', ha='center', va='center', 
                        fontsize=7, fontweight='bold')
    
    def _draw_orders(self, env):
        """Draw pending orders"""
        for order_id, order_info in env.orders.items():
            if not order_info["delivered"]:
                cust_pos = env.customer_positions[order_info["customer"]]
                
                # Draw circle around customer for pending order
                circle = patches.Circle((cust_pos[1], cust_pos[0]), 0.5, 
                                       fill=False, edgecolor=self.colors['order'], 
                                       linewidth=2, linestyle='--', zorder=2)
                self.ax.add_patch(circle)
    
    def _add_info_panel(self, episode: int, step: int, reward: float, env):
        """Add information panel"""
        # Count deliveries
        delivered_count = sum(1 for order in env.orders.values() if order["delivered"])
        total_orders = len(env.orders)
        
        # Create info text
        info_text = (
            f"Episode: {episode}\n"
            f"Step: {step}/{env.max_steps}\n"
            f"Reward: {reward:.2f}\n"
            f"Delivered: {delivered_count}/{total_orders}\n"
            f"Agent1 Pos: ({env.agent1_pos[0]}, {env.agent1_pos[1]})\n"
            f"Agent2 Pos: ({env.agent2_pos[0]}, {env.agent2_pos[1]})"
        )
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=props)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor=self.colors['agent1'], label='Agent 1'),
            patches.Patch(facecolor=self.colors['agent2'], label='Agent 2'),
            patches.Patch(facecolor=self.colors['restaurant'], label='Restaurant'),
            patches.Patch(facecolor=self.colors['customer'], label='Customer'),
            patches.Patch(facecolor='white', edgecolor=self.colors['order'], 
                         linestyle='--', linewidth=2, label='Pending Order')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def close(self):
        """Close figure"""
        if self.fig:
            plt.close(self.fig)
    
    def create_video(self, output_path: str = "logs/training_video.mp4", fps: int = 5):
        """
        Create video from rendered frames
        
        Args:
            output_path: Path to save video
            fps: Frames per second
        """
        try:
            import cv2
            
            # Get all frame files
            frame_files = sorted(self.output_dir.glob("episode_*.png"))
            
            if not frame_files:
                print("❌ No frames found to create video")
                return
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                out.write(frame)
            
            out.release()
            print(f"✓ Video created: {output_path}")
        
        except ImportError:
            print("⚠️ OpenCV not installed. Install with: pip install opencv-python")
