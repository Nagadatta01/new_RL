"""
[STEP 2] HYPERPARAMETER IDENTIFICATION & RATIONALE (2 Marks)
"""

import json
from pathlib import Path


def step2_hyperparameter_rationale():
    """Step 2: Hyperparameter Identification & Rationale"""
    
    print("\n" + "="*80)
    print("[STEP 2] HYPERPARAMETER IDENTIFICATION & RATIONALE (2 Marks)")
    print("="*80)
    
    parameters = {
        "learning_rate": {
            "selected": 0.001,
            "range": [0.0001, 0.001, 0.01],
            "rationale": (
                "Controls the step size for weight updates in DQN.\n"
                "  • 0.0001: Too low - slow convergence\n"
                "  • 0.001: SELECTED - standard, balances speed & stability\n"
                "  • 0.01: Too high - unstable"
            )
        },
        "gamma": {
            "selected": 0.95,
            "range": [0.90, 0.95, 0.99],
            "rationale": (
                "Discount factor for future rewards.\n"
                "  • 0.90: Myopic (short-term focus)\n"
                "  • 0.95: SELECTED - balanced for delivery tasks\n"
                "  • 0.99: Far-sighted (long-term focus)"
            )
        },
        "buffer_size": {
            "selected": 10000,
            "range": [1000, 10000, 50000],
            "rationale": (
                "Experience replay buffer capacity.\n"
                "  • 1000: Small, insufficient diversity\n"
                "  • 10000: SELECTED - good balance\n"
                "  • 50000: Large, high memory"
            )
        },
        "batch_size": {
            "selected": 32,
            "range": [16, 32, 64],
            "rationale": (
                "Samples per gradient update.\n"
                "  • 16: Small, noisier gradients\n"
                "  • 32: SELECTED - balanced\n"
                "  • 64: Large, smoother but slower"
            )
        },
        "target_update_freq": {
            "selected": 100,
            "range": [50, 100, 500],
            "rationale": (
                "Target Q-network update interval.\n"
                "  • 50: Frequent, may be unstable\n"
                "  • 100: SELECTED - moderate, balanced\n"
                "  • 500: Rare, slow adaptation"
            )
        }
    }
    
    for param_name, param_info in parameters.items():
        print(f"\n{'='*70}")
        print(f"Parameter: {param_name.upper()}")
        print(f"{'='*70}")
        print(f"Selected Value: {param_info['selected']}")
        print(f"Tuning Range: {param_info['range']}")
        print(f"\nRationale:")
        print(param_info['rationale'])
    
    results_dir = Path("results/step2_rationale")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = {param: {k: v for k, v in info.items() if k != "rationale"} 
                 for param, info in parameters.items()}
    
    with open(results_dir / "hyperparameter_rationale.json", "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Rationale saved to: {results_dir}/hyperparameter_rationale.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    step2_hyperparameter_rationale()
