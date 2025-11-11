"""
[STEP 5] MODEL DISCUSSION AND CONCLUSION (2 Marks)
Compare baseline vs tuned
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def step5_discussion_conclusion():
    """Step 5: Discussion & Conclusion"""
    
    print("\n" + "="*80)
    print("[STEP 5] MODEL DISCUSSION AND CONCLUSION (2 Marks)")
    print("="*80)
    
    results_dir = Path("results/step5_discussion")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open("results/step3_baseline/baseline_results.json") as f:
        baseline = json.load(f)
    
    with open("results/step3_ablation/ablation_results.json") as f:
        ablation = json.load(f)
    
    with open("results/step4_evaluation/test_results.json") as f:
        test = json.load(f)
    
    baseline_config = baseline["configuration"]
    tuned_config = ablation["best_config"]
    baseline_reward = baseline["aggregate_results"]["mean_reward"]
    
    # Comparison table
    print("\n📊 COMPARISON: BASELINE vs TUNED\n")
    
    comparison = {
        "Parameter": ["Learning Rate", "Discount Factor", "Buffer Size", "Batch Size", "Target Update"],
        "Baseline": [baseline_config["learning_rate"], baseline_config["gamma"],
                    baseline_config["buffer_size"], baseline_config["batch_size"],
                    baseline_config["target_update_freq"]],
        "Tuned": [tuned_config["learning_rate"], tuned_config["gamma"],
                 tuned_config["buffer_size"], tuned_config["batch_size"],
                 tuned_config["target_update_freq"]]
    }
    
    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))
    
    print("\n\n📈 TEST PERFORMANCE:\n")
    print(f"  Unseen Episodes: Reward={test['unseen']['avg_reward']:.2f}, Success={test['unseen']['success_rate']*100:.0f}%")
    print(f"  Altered Env: Reward={test['altered']['avg_reward']:.2f}, Success={test['altered']['success_rate']*100:.0f}%")
    print(f"  Noisy Obs: Reward={test['noisy']['avg_reward']:.2f}, Success={test['noisy']['success_rate']*100:.0f}%")
    
    # Create final plot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('DQN Project - Complete Analysis', fontsize=16, fontweight='bold')
    
    # Table
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    table_data = [
        ["Parameter", "Baseline", "Tuned", "Changed"],
        ["Learning Rate", str(baseline_config["learning_rate"]), str(tuned_config["learning_rate"]), 
         "✓" if baseline_config["learning_rate"] != tuned_config["learning_rate"] else "✗"],
        ["Discount Factor", str(baseline_config["gamma"]), str(tuned_config["gamma"]), 
         "✓" if baseline_config["gamma"] != tuned_config["gamma"] else "✗"],
        ["Buffer Size", str(baseline_config["buffer_size"]), str(tuned_config["buffer_size"]), 
         "✓" if baseline_config["buffer_size"] != tuned_config["buffer_size"] else "✗"],
        ["Batch Size", str(baseline_config["batch_size"]), str(tuned_config["batch_size"]), 
         "✓" if baseline_config["batch_size"] != tuned_config["batch_size"] else "✗"],
        ["Target Update", str(baseline_config["target_update_freq"]), str(tuned_config["target_update_freq"]), 
         "✓" if baseline_config["target_update_freq"] != tuned_config["target_update_freq"] else "✗"]
    ]
    
    table = ax1.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, 6):
        table[(i, 0)].set_facecolor('#e8f4f8')
        table[(i, 1)].set_facecolor('#ffe8e8')
        table[(i, 2)].set_facecolor('#e8ffe8')
        table[(i, 3)].set_facecolor('#fff8e8')
    
    ax1.set_title('Hyperparameter Configuration Comparison', fontsize=12, fontweight='bold', pad=10)
    
    # Test results
    ax2 = fig.add_subplot(gs[1, 0])
    scenarios = ["Unseen", "Altered", "Noisy"]
    rewards = [test['unseen']['avg_reward'], test['altered']['avg_reward'], test['noisy']['avg_reward']]
    bars = ax2.bar(scenarios, rewards, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Avg Reward', fontweight='bold')
    ax2.set_title('Test Scenario Rewards', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Success rates
    ax3 = fig.add_subplot(gs[1, 1])
    success_rates = [test['unseen']['success_rate']*100, test['altered']['success_rate']*100, test['noisy']['success_rate']*100]
    bars = ax3.bar(scenarios, success_rates, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Success Rate (%)', fontweight='bold')
    ax3.set_ylim([0, 110])
    ax3.set_title('Test Scenario Success Rates', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(results_dir / "final_analysis_discussion.png", dpi=150, bbox_inches='tight')
    
    # Save summary
    with open(results_dir / "discussion_summary.json", "w") as f:
        json.dump({
            "baseline_config": baseline_config,
            "tuned_config": tuned_config,
            "baseline_reward": float(baseline_reward),
            "test_results": test
        }, f, indent=2)
    
    print(f"\n✓ Saved: {results_dir}/final_analysis_discussion.png")
    print(f"✓ Saved: {results_dir}/discussion_summary.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    step5_discussion_conclusion()
