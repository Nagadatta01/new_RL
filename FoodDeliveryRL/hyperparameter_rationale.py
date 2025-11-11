"""
[2] HYPERPARAMETER IDENTIFICATION & RATIONALE (2 Marks)
- Critical DQN parameters identified
- Justification with research rationale
"""

import json
from pathlib import Path


def hyperparameter_rationale():
    """Document hyperparameter selections with rationale"""
    
    print("\n" + "="*80)
    print("[2] HYPERPARAMETER IDENTIFICATION & RATIONALE")
    print("="*80)
    
    hyperparameters = {
        "Learning Rate (α)": {
            "value": 0.001,
            "range": [0.0001, 0.001, 0.01],
            "rationale": (
                "Controls DQN weight updates. 0.001 is standard for neural networks.\n"
                "• Too high (0.01): Unstable, overshoots optimal weights\n"
                "• Too low (0.0001): Slow convergence, may get stuck\n"
                "• Sweet spot (0.001): Balances speed and stability"
            )
        },
        "Discount Factor (γ)": {
            "value": 0.95,
            "range": [0.9, 0.95, 0.99],
            "rationale": (
                "Determines importance of future rewards. γ = 0.95 chosen.\n"
                "• γ = 0.9: Myopic agent (short-term focus)\n"
                "• γ = 0.95: Balance (current choice) - good for delivery tasks\n"
                "• γ = 0.99: Far-sighted agent (long-term focus)"
            )
        },
        "Epsilon Start (ε_start)": {
            "value": 1.0,
            "range": [0.5, 1.0],
            "rationale": (
                "Initial exploration rate. ε_start = 1.0 chosen.\n"
                "• 1.0 = Maximum exploration: Agent explores all actions randomly\n"
                "• Justified: Agent needs to discover environment structure"
            )
        },
        "Epsilon End (ε_end)": {
            "value": 0.1,
            "range": [0.01, 0.05, 0.1],
            "rationale": (
                "Minimum exploration rate after decay. ε_end = 0.1 chosen.\n"
                "• 0.01: Very low exploration (mostly exploitation)\n"
                "• 0.1: More exploration (current choice) - prevents premature convergence\n"
                "• Justification: Food delivery task benefits from continued exploration"
            )
        },
        "Epsilon Decay": {
            "value": 0.995,
            "range": [0.99, 0.995, 0.999],
            "rationale": (
                "Controls exploration → exploitation transition. 0.995 chosen.\n"
                "• 0.99: Fast decay (reaches ε_end quickly)\n"
                "• 0.995: Moderate decay (current choice) - balanced\n"
                "• 0.999: Slow decay (long exploration phase)"
            )
        },
        "Replay Buffer Size": {
            "value": 10000,
            "range": [1000, 5000, 10000],
            "rationale": (
                "Stores past experiences for learning. 10000 chosen.\n"
                "• 1000: Small buffer, loses old experiences quickly\n"
                "• 10000: Large buffer (current choice) - diverse samples, stable learning"
            )
        },
        "Batch Size": {
            "value": 32,
            "range": [16, 32, 64],
            "rationale": (
                "Number of samples per training step. 32 chosen.\n"
                "• 16: Small batch - noisier gradient updates\n"
                "• 32: Medium batch (current choice) - balanced\n"
                "• 64: Large batch - smoother gradients, slower training"
            )
        },
        "Target Network Update": {
            "value": 100,
            "range": [50, 100, 500],
            "rationale": (
                "Update target network every N training steps. 100 chosen.\n"
                "• 50: Frequent updates (may cause instability)\n"
                "• 100: Moderate (current choice) - balances stability & convergence\n"
                "• 500: Infrequent (slow adaptation)"
            )
        }
    }
    
    for param_name, param_info in hyperparameters.items():
        print(f"\n{'='*60}")
        print(f"Parameter: {param_name}")
        print(f"{'='*60}")
        print(f"Selected Value: {param_info['value']}")
        print(f"Tuning Range: {param_info['range']}")
        print(f"\nRationale:\n{param_info['rationale']}")
    
    # Save rationale
    rationale_dir = Path("analysis")
    rationale_dir.mkdir(exist_ok=True)
    
    with open(rationale_dir / "hyperparameter_rationale.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)
    
    print(f"\n✓ Hyperparameter rationale saved to: {rationale_dir}/hyperparameter_rationale.json")
    print("\n" + "="*80 + "\n")
    
    return hyperparameters


if __name__ == "__main__":
    hyperparameter_rationale()
