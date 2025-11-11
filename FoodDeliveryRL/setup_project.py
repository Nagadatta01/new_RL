import os
from pathlib import Path

# Define directory structure
directories = [
    "environment",
    "models",
    "training",
    "utils",
    "visualization",
    "logs",
    "logs/baseline",
    "logs/tuned",
    "logs/tuning",
    "data"
]

# Create all directories
for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {directory}")

# Create __init__.py files for packages
packages = ["environment", "models", "training", "utils", "visualization"]

for package in packages:
    init_file = Path(package) / "__init__.py"
    init_file.touch()
    print(f"✓ Created: {init_file}")

print("\n✓ Project structure setup complete!")
