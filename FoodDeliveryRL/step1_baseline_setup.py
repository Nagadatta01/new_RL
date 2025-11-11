"""
[STEP 1] EXPERIMENTAL SETUP AND BASELINE CONFIGURATION (2 Marks)
"""

import torch
import numpy as np
from pathlib import Path
import json
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def step1_baseline_setup():
    """Step 1: Experimental Setup and Baseline Configuration"""
    
    print("\n" + "="*80)
    print("[STEP 1] EXPERIMENTAL SETUP & BASELINE CONFIGURATION (2 Marks)")
    print("="*80)
    
    print("\n📍 ENVIRONMENT DEFINITION:")
    env = RealisticDeliveryEnvironment(grid_size=15, num_restaurants=3, num_customers=5)
    
    print(f"\n  ✓ Environment Type: Custom Food Delivery RL")
    print(f"  ✓ Grid Size: 15×15 cells")
    print(f"  ✓ Restaurants: 3 (randomly placed)")
    print(f"  ✓ Customers: 5 (randomly placed)")
    
    print(f"\n  📊 STATE SPACE (5D Vector):")
    print(f"     • agent_x: Agent X position (0-15)")
    print(f"     • agent_y: Agent Y position (0-15)")
    print(f"     • carrying: 0=empty, 1=carrying package")
    print(f"     • target_x: Target location X")
    print(f"     • target_y: Target location Y")
    print(f"     → Total State Dimension: {int(env.state_dim)}")
    
    print(f"\n  🎮 ACTION SPACE (6 Discrete Actions):")
    print(f"     • Action 0: Move Up (y-1)")
    print(f"     • Action 1: Move Down (y+1)")
    print(f"     • Action 2: Move Left (x-1)")
    print(f"     • Action 3: Move Right (x+1)")
    print(f"     • Action 4: Pickup from Restaurant")
    print(f"     • Action 5: Delivery to Customer")
    print(f"     → Total Action Dimension: {int(env.action_space.n)}")
    
    print(f"\n  💰 REWARD DESIGN:")
    print(f"     • Collision: -2.0 (hit obstacle)")
    print(f"     • Movement: (distance_improvement × 0.5)")
    print(f"     • Pickup Success: +20.0")
    print(f"     • Delivery Success: +100.0")
    print(f"     • Episode Complete: +200.0 (all 5 delivered)")
    
    print(f"\n🧠 BASELINE DQN MODEL INITIALIZATION:")
    
    baseline_config = {
        "learning_rate": 0.001,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_freq": 100
    }
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        **baseline_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\n  ✓ Network Architecture: 5 → 128 → 128 → 128 → 6")
    print(f"  ✓ Optimizer: Adam")
    print(f"  ✓ Loss Function: MSE (Mean Squared Error)")
    print(f"  ✓ Device: {str(agent.device).upper()}")
    
    print(f"\n  📋 BASELINE HYPERPARAMETERS:")
    for param, value in baseline_config.items():
        print(f"     • {param}: {value}")
    
    print(f"\n✅ REPRODUCIBILITY CONFIGURATION:")
    print(f"  ✓ Random Seed (torch): {SEED}")
    print(f"  ✓ Random Seed (numpy): {SEED}")
    print(f"  ✓ All random operations are deterministic")
    
    config_dir = Path("results/step1_baseline")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_data = {
        "environment": {
            "grid_size": 15,
            "num_restaurants": 3,
            "num_customers": 5,
            "state_dim": int(env.state_dim),
            "action_dim": int(env.action_space.n),
            "max_steps": int(env.max_steps)
        },
        "baseline_dqn": baseline_config,
        "seed": SEED,
        "device": str(agent.device)
    }
    
    with open(config_dir / "baseline_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n✓ Configuration saved to: {config_dir}/baseline_config.json")
    print("\n" + "="*80 + "\n")
    
    env.close()
    return baseline_config


if __name__ == "__main__":
    config = step1_baseline_setup()
