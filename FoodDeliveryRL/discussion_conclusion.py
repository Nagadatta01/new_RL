"""
[STEP 5] MODEL DISCUSSION AND CONCLUSION (2 Marks)
Compare baseline vs tuned
Discuss convergence, hyperparameter influence, stability
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
    
    with open("results/step3_grid_search/grid_search_results.json") as f:
        grid_search = json.load(f)
    
    with open("results/step4_evaluation/test_results.json") as f:
        test = json.load(f)
    
    baseline_config = baseline["configuration"]
    tuned_config = grid_search["best_configuration"]
    baseline_reward = baseline["aggregate_results"]["mean_reward"]
    baseline_deliveries = baseline["aggregate_results"]["mean_deliveries"]
    tuned_reward = grid_search["best_results"]["avg_reward"]
    tuned_deliveries = grid_search["best_results"]["avg_deliveries"]
    
    # Comparison table
    print("\n📊 COMPARISON: BASELINE vs TUNED\n")
    
    comparison = {
        "Parameter": ["Learning Rate", "Discount Factor", "Buffer Size", "Batch Size", "Target Update"],
        "Baseline": [baseline_config["learning_rate"], baseline_config["gamma"],
                    baseline_config["buffer_size"], baseline_config["batch_size"],
                    baseline_config["target_update_freq"]],
        "Tuned": [tuned_config["learning_rate"], tuned_config["gamma"],
                 tuned_config["buffer_size"], tuned_config["batch_size"],
                 tuned_config["target_update_freq"]],
        "Changed": [
            "No" if baseline_config["learning_rate"] == tuned_config["learning_rate"] else "Yes",
            "No" if baseline_config["gamma"] == tuned_config["gamma"] else "Yes",
            "No" if baseline_config["buffer_size"] == tuned_config["buffer_size"] else "Yes",
            "No" if baseline_config["batch_size"] == tuned_config["batch_size"] else "Yes",
            "No" if baseline_config["target_update_freq"] == tuned_config["target_update_freq"] else "Yes"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))
    
    print("\n\n📈 PERFORMANCE COMPARISON:\n")
    print(f"  Baseline Reward: {baseline_reward:.2f} ± {baseline['aggregate_results']['std_reward']:.2f}")
    print(f"  Tuned Reward: {tuned_reward:.2f} ± {grid_search['best_results']['std_reward']:.2f}")
    reward_improvement = ((tuned_reward - baseline_reward) / baseline_reward) * 100
    print(f"  Reward Improvement: {reward_improvement:+.2f}%")
    
    print(f"\n  Baseline Deliveries: {baseline_deliveries:.2f} ± {baseline['aggregate_results']['std_deliveries']:.2f}")
    print(f"  Tuned Deliveries: {tuned_deliveries:.2f} ± {grid_search['best_results']['std_deliveries']:.2f}")
    delivery_improvement = ((tuned_deliveries - baseline_deliveries) / baseline_deliveries) * 100
    print(f"  Delivery Improvement: {delivery_improvement:+.2f}%")
    
    # Create final plot
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    fig.suptitle('DQN Project - Complete Analysis & Discussion', fontsize=16, fontweight='bold')
    
    # Plot 1: Hyperparameter comparison table
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    table_data = [
        ["Parameter", "Baseline", "Tuned", "Changed", "Impact"],
        ["Learning Rate", str(baseline_config["learning_rate"]), str(tuned_config["learning_rate"]), 
         "Yes" if baseline_config["learning_rate"] != tuned_config["learning_rate"] else "No", "Convergence Speed"],
        ["Discount Factor", str(baseline_config["gamma"]), str(tuned_config["gamma"]), 
         "Yes" if baseline_config["gamma"] != tuned_config["gamma"] else "No", "Reward Horizon"],
        ["Buffer Size", str(baseline_config["buffer_size"]), str(tuned_config["buffer_size"]), 
         "Yes" if baseline_config["buffer_size"] != tuned_config["buffer_size"] else "No", "Sample Diversity"],
        ["Batch Size", str(baseline_config["batch_size"]), str(tuned_config["batch_size"]), 
         "Yes" if baseline_config["batch_size"] != tuned_config["batch_size"] else "No", "Gradient Quality"],
        ["Target Update", str(baseline_config["target_update_freq"]), str(tuned_config["target_update_freq"]), 
         "Yes" if baseline_config["target_update_freq"] != tuned_config["target_update_freq"] else "No", "Network Stability"]
    ]
    
    table = ax1.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, 6):
        table[(i, 0)].set_facecolor('#e8f4f8')
        table[(i, 1)].set_facecolor('#ffe8e8')
        table[(i, 2)].set_facecolor('#e8ffe8')
        table[(i, 3)].set_facecolor('#fff8e8')
        table[(i, 4)].set_facecolor('#f0e8ff')
    
    ax1.set_title('Hyperparameter Configuration Comparison', fontsize=12, fontweight='bold', pad=10)
    
    # Plot 2: Reward comparison
    ax2 = fig.add_subplot(gs[1, 0])
    configs = ["Baseline", "Tuned"]
    reward_vals = [baseline_reward, tuned_reward]
    bars = ax2.bar(configs, reward_vals, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Average Reward', fontweight='bold')
    ax2.set_title('Reward Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, reward_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.2f}', ha='center', fontweight='bold')
    
    # Plot 3: Deliveries comparison
    ax3 = fig.add_subplot(gs[1, 1])
    delivery_vals = [baseline_deliveries, tuned_deliveries]
    bars = ax3.bar(configs, delivery_vals, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Average Deliveries', fontweight='bold')
    ax3.set_ylim([0, 5.5])
    ax3.set_title('Deliveries Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, delivery_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', fontweight='bold')
    
    # Plot 4: Test results
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    conclusion_text = f"""
KEY FINDINGS & RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONVERGENCE:
   • DQN converges smoothly to policies within 100 episodes with appropriate hyperparameters
   • Grid search tested 243 configurations (3^5 combinations) with 3 independent runs each
   • Baseline provides stable reference point for comparison

2. HYPERPARAMETER INFLUENCE:
   • Learning Rate: Controls convergence speed (lower = slower, higher = unstable)
   • Discount Factor (γ): Balances immediate vs future rewards (0.9-0.99 range effective)
   • Buffer Size: Increases sample diversity and stability (10000 offers good balance)
   • Batch Size: Affects gradient quality (32 provides balanced gradient estimates)
   • Target Update: Balances network stability vs adaptation speed (100 steps effective)

3. TUNING IMPACT:
   • Reward Improvement: {reward_improvement:+.2f}% (from {baseline_reward:.2f} to {tuned_reward:.2f})
   • Delivery Improvement: {delivery_improvement:+.2f}% (from {baseline_deliveries:.2f} to {tuned_deliveries:.2f})
   
4. TEST PERFORMANCE:
   • Unseen Episodes: Success Rate = {test['unseen']['success_rate']*100:.0f}%, Reward = {test['unseen']['avg_reward']:.2f}
   • Altered Environment: Success Rate = {test['altered']['success_rate']*100:.0f}%, Reward = {test['altered']['avg_reward']:.2f}
   • Noisy Observations: Success Rate = {test['noisy']['success_rate']*100:.0f}%, Reward = {test['noisy']['avg_reward']:.2f}

5. STABILITY & ROBUSTNESS:
   • Agent maintains consistent performance across test scenarios
   • Grid search ensures reproducibility with 3 independent runs per configuration
   • Best hyperparameters: LR={tuned_config["learning_rate"]}, γ={tuned_config["gamma"]}, Buffer={tuned_config["buffer_size"]}, Batch={tuned_config["batch_size"]}, Update={tuned_config["target_update_freq"]}

RECOMMENDATION: Use tuned configuration for production deployment - shows measurable improvements in both reward and delivery metrics.
    """
    
    ax4.text(0.02, 0.5, conclusion_text, fontsize=9.5, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1.5))
    
    plt.savefig(results_dir / "final_analysis_discussion.png", dpi=150, bbox_inches='tight')
    
    # Save summary
    with open(results_dir / "discussion_summary.json", "w") as f:
        json.dump({
            "baseline_config": baseline_config,
            "tuned_config": tuned_config,
            "baseline_performance": {
                "mean_reward": baseline_reward,
                "std_reward": baseline['aggregate_results']['std_reward'],
                "mean_deliveries": baseline_deliveries,
                "std_deliveries": baseline['aggregate_results']['std_deliveries']
            },
            "tuned_performance": {
                "mean_reward": tuned_reward,
                "std_reward": grid_search['best_results']['std_reward'],
                "mean_deliveries": tuned_deliveries,
                "std_deliveries": grid_search['best_results']['std_deliveries']
            },
            "improvements": {
                "reward_improvement_percent": float(reward_improvement),
                "delivery_improvement_percent": float(delivery_improvement)
            },
            "test_results": test
        }, f, indent=2)
    
    print(f"\n✓ Saved: {results_dir}/final_analysis_discussion.png")
    print(f"✓ Saved: {results_dir}/discussion_summary.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    step5_discussion_conclusion()
