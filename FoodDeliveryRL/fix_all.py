import os
import re

files_to_fix = [
    "environment/food_delivery_env.py",
    "environment/multi_agent_env.py",
]

# Fix imports in environment files
for filepath in files_to_fix:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace gym with gymnasium
        content = content.replace(
            "import gym\nfrom gym import spaces",
            "import gymnasium as gym\nfrom gymnasium import spaces"
        )
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed: {filepath}")

# Fix hyperparameter_tuning.py
hp_file = "training/hyperparameter_tuning.py"
if os.path.exists(hp_file):
    with open(hp_file, 'r') as f:
        content = f.read()
    
    # Add missing import
    if "from typing import Dict" not in content:
        content = content.replace(
            "from itertools import product\nimport sys",
            "from itertools import product\nfrom typing import Dict\nimport sys"
        )
    
    with open(hp_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed: {hp_file}")

print("\n✓ All files fixed!")
