"""
[STEP 5] COMPREHENSIVE COMPARISON PLOTS
- Ablation study plots (one parameter at a time)
- Comparison across all hyperparameters
- Summary table
- Training curves
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_ablation_plots():
    """Create all comparison plots for ablation study"""
    
    # Load results
    results_file = Path("results/ablation_study/ablation_study_results.json")
    
    if not results_file.exists():
        print("❌ Results file not found! Run ablation_study_training.py first!")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    best_config = results["best_config"]
    
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("[STEP 5] GENERATING COMPARISON PLOTS")
    print("="*80)
    
    # ===== PLOT 1: Learning Rate Ablation =====
    print("\n  Creating Plot 1: Learning Rate Sensitivity...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    lrs = sorted([float(k) for k in results["learning_rate"].keys()])
    rewards = [results["learning_rate"][str(lr)]["avg_reward"] for lr in lrs]
    stds = [results["learning_rate"][str(lr)]["std_reward"] for lr in lrs]
    
    ax1.errorbar(range(len(lrs)), rewards, yerr=stds, fmt='o-', linewidth=2, markersize=10, capsize=5)
    ax1.set_xticks(range(len(lrs)))
    ax1.set_xticklabels([f"{lr:.4f}" for lr in lrs])
    ax1.set_xlabel('Learning Rate (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Final Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Rate Ablation Study', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    deliveries = [results["learning_rate"][str(lr)]["avg_deliveries"] for lr in lrs]
    ax2.bar(range(len(lrs)), deliveries, color='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax2.set_xticks(range(len(lrs)))
    ax2.set_xticklabels([f"{lr:.4f}" for lr in lrs])
    ax2.set_xlabel('Learning Rate (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Deliveries', fontsize=12, fontweight='bold')
    ax2.set_title('Deliveries Completed', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 5])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "01_learning_rate_ablation.png", dpi=150)
    print("  ✓ Saved: 01_learning_rate_ablation.png")
    
    # ===== PLOT 2: Gamma Ablation =====
    print("  Creating Plot 2: Discount Factor Sensitivity...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    gammas = sorted([float(k) for k in results["gamma"].keys()])
    rewards = [results["gamma"][str(g)]["avg_reward"] for g in gammas]
    stds = [results["gamma"][str(g)]["std_reward"] for g in gammas]
    
    ax1.errorbar(range(len(gammas)), rewards, yerr=stds, fmt='s-', linewidth=2, markersize=10, capsize=5, color='orange')
    ax1.set_xticks(range(len(gammas)))
    ax1.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax1.set_xlabel('Discount Factor (γ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Final Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Discount Factor Ablation Study', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    deliveries = [results["gamma"][str(g)]["avg_deliveries"] for g in gammas]
    ax2.bar(range(len(gammas)), deliveries, color='lightblue', edgecolor='darkblue', linewidth=2)
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels([f"{g:.2f}" for g in gammas])
    ax2.set_xlabel('Discount Factor (γ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Deliveries', fontsize=12, fontweight='bold')
    ax2.set_title('Deliveries Completed', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 5])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "02_gamma_ablation.png", dpi=150)
    print("  ✓ Saved: 02_gamma_ablation.png")
    
    # ===== PLOT 3: Buffer Size Ablation =====
    print("  Creating Plot 3: Buffer Size Sensitivity...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    buffers = sorted([int(k) for k in results["buffer_size"].keys()])
    rewards = [results["buffer_size"][str(b)]["avg_reward"] for b in buffers]
    stds = [results["buffer_size"][str(b)]["std_reward"] for b in buffers]
    
    ax1.errorbar(range(len(buffers)), rewards, yerr=stds, fmt='^-', linewidth=2, markersize=10, capsize=5, color='purple')
    ax1.set_xticks(range(len(buffers)))
    ax1.set_xticklabels([f"{b//1000}K" for b in buffers])
    ax1.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Final Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Buffer Size Ablation Study', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    deliveries = [results["buffer_size"][str(b)]["avg_deliveries"] for b in buffers]
    ax2.bar(range(len(buffers)), deliveries, color='lightyellow', edgecolor='goldenrod', linewidth=2)
    ax2.set_xticks(range(len(buffers)))
    ax2.set_xticklabels([f"{b//1000}K" for b in buffers])
    ax2.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Deliveries', fontsize=12, fontweight='bold')
    ax2.set_title('Deliveries Completed', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 5])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "03_buffer_size_ablation.png", dpi=150)
    print("  ✓ Saved: 03_buffer_size_ablation.png")
    
    # ===== PLOT 4: Batch Size Ablation =====
    print("  Creating Plot 4: Batch Size Sensitivity...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    batches = sorted([int(k) for k in results["batch_size"].keys()])
    rewards = [results["batch_size"][str(b)]["avg_reward"] for b in batches]
    stds = [results["batch_size"][str(b)]["std_reward"] for b in batches]
    
    ax1.errorbar(range(len(batches)), rewards, yerr=stds, fmt='D-', linewidth=2, markersize=10, capsize=5, color='red')
    ax1.set_xticks(range(len(batches)))
    ax1.set_xticklabels(batches)
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Final Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Batch Size Ablation Study', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    deliveries = [results["batch_size"][str(b)]["avg_deliveries"] for b in batches]
    ax2.bar(range(len(batches)), deliveries, color='lightcoral', edgecolor='darkred', linewidth=2)
    ax2.set_xticks(range(len(batches)))
    ax2.set_xticklabels(batches)
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Deliveries', fontsize=12, fontweight='bold')
    ax2.set_title('Deliveries Completed', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 5])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "04_batch_size_ablation.png", dpi=150)
    print("  ✓ Saved: 04_batch_size_ablation.png")
    
    # ===== PLOT 5: Target Update Ablation =====
    print("  Creating Plot 5: Target Update Frequency Sensitivity...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    updates = sorted([int(k) for k in results["target_update_freq"].keys()])
    rewards = [results["target_update_freq"][str(u)]["avg_reward"] for u in updates]
    stds = [results["target_update_freq"][str(u)]["std_reward"] for u in updates]
    
    ax1.errorbar(range(len(updates)), rewards, yerr=stds, fmt='o-', linewidth=2, markersize=10, capsize=5, color='green')
    ax1.set_xticks(range(len(updates)))
    ax1.set_xticklabels(updates)
    ax1.set_xlabel('Target Network Update Interval', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Final Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Target Update Frequency Ablation Study', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    deliveries = [results["target_update_freq"][str(u)]["avg_deliveries"] for u in updates]
    ax2.bar(range(len(updates)), deliveries, color='lightseagreen', edgecolor='darkslategray', linewidth=2)
    ax2.set_xticks(range(len(updates)))
    ax2.set_xticklabels(updates)
    ax2.set_xlabel('Target Network Update Interval', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Deliveries', fontsize=12, fontweight='bold')
    ax2.set_title('Deliveries Completed', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 5])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "05_target_update_ablation.png", dpi=150)
    print("  ✓ Saved: 05_target_update_ablation.png")
    
    # ===== PLOT 6: Summary Comparison Table =====
    print("  Creating Plot 6: Summary Table...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    table_data = [
        ["Hyperparameter", "Best Value", "Reward", "Deliveries/5"],
        ["Learning Rate (α)", f"{best_config['learning_rate']}", 
         f"{results['learning_rate'][str(best_config['learning_rate'])]['avg_reward']:.2f}", 
         f"{results['learning_rate'][str(best_config['learning_rate'])]['avg_deliveries']:.2f}"],
        ["Discount Factor (γ)", f"{best_config['gamma']}", 
         f"{results['gamma'][str(best_config['gamma'])]['avg_reward']:.2f}", 
         f"{results['gamma'][str(best_config['gamma'])]['avg_deliveries']:.2f}"],
        ["Buffer Size", f"{best_config['buffer_size']}", 
         f"{results['buffer_size'][str(best_config['buffer_size'])]['avg_reward']:.2f}", 
         f"{results['buffer_size'][str(best_config['buffer_size'])]['avg_deliveries']:.2f}"],
        ["Batch Size", f"{best_config['batch_size']}", 
         f"{results['batch_size'][str(best_config['batch_size'])]['avg_reward']:.2f}", 
         f"{results['batch_size'][str(best_config['batch_size'])]['avg_deliveries']:.2f}"],
        ["Target Update Freq", f"{best_config['target_update_freq']}", 
         f"{results['target_update_freq'][str(best_config['target_update_freq'])]['avg_reward']:.2f}", 
         f"{results['target_update_freq'][str(best_config['target_update_freq'])]['avg_deliveries']:.2f}"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best config rows
    for i in range(1, 6):
        table[(i, 0)].set_facecolor('#e8f4f8')
        table[(i, 1)].set_facecolor('#c8e6c9')
        table[(i, 2)].set_facecolor('#fff9c4')
        table[(i, 3)].set_facecolor('#ffccbc')
    
    plt.title('Best Hyperparameter Configuration (Ablation Study Results)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(plots_dir / "06_summary_table.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: 06_summary_table.png")
    
    # ===== PLOT 7: All Parameters Comparison =====
    print("  Creating Plot 7: Overall Comparison...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    params = ['Learning Rate\n(0.001)', 'Discount Factor\n(0.95)', 'Buffer Size\n(10000)', 
              'Batch Size\n(32)', 'Target Update\n(100)']
    best_rewards = [
        results['learning_rate'][str(best_config['learning_rate'])]['avg_reward'],
        results['gamma'][str(best_config['gamma'])]['avg_reward'],
        results['buffer_size'][str(best_config['buffer_size'])]['avg_reward'],
        results['batch_size'][str(best_config['batch_size'])]['avg_reward'],
        results['target_update_freq'][str(best_config['target_update_freq'])]['avg_reward']
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(params, best_rewards, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Average Final Reward', fontsize=12, fontweight='bold')
    ax.set_title('Best Performance for Each Hyperparameter', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, reward in zip(bars, best_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{reward:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "07_all_parameters_comparison.png", dpi=150)
    print("  ✓ Saved: 07_all_parameters_comparison.png")
    
    print(f"\n✓ All plots saved to: {plots_dir}/")
    print("\n" + "="*80)


if __name__ == "__main__":
    create_ablation_plots()
