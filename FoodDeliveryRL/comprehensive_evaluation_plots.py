"""
Complete evaluation with:
- Training curves (Reward, Loss, Epsilon)
- 3 Scenarios with comparison tables
- Hyperparameter sensitivity analysis
- All plots saved automatically
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from environment.realistic_delivery_env import RealisticDeliveryEnvironment
from models.dqn_agent import DQNAgent

SEED = 42

def create_comprehensive_plots(results, best_configs):
    """Generate all comparison plots"""
    
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating Comprehensive Plots...")
    
    # ===== PLOT 1: Hyperparameter Sensitivity Analysis =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for sensitivity
    lr_performance = {}
    gamma_performance = {}
    epsilon_performance = {}
    
    for config_key, scenario_data in results.items():
        parts = config_key.split("_")
        lr = float(parts[0].split("=")[1])
        gamma = float(parts[1].split("=")[1])
        eps = float(parts[2].split("=")[1])
        
        # Average across scenarios
        avg_reward = np.mean([scenario_data[s]["avg_final_reward"] for s in scenario_data.keys()])
        
        if lr not in lr_performance:
            lr_performance[lr] = []
        lr_performance[lr].append(avg_reward)
        
        if gamma not in gamma_performance:
            gamma_performance[gamma] = []
        gamma_performance[gamma].append(avg_reward)
        
        if eps not in epsilon_performance:
            epsilon_performance[eps] = []
        epsilon_performance[eps].append(avg_reward)
    
    # Plot 1: Learning Rate Sensitivity
    lrs = sorted(lr_performance.keys())
    lr_means = [np.mean(lr_performance[lr]) for lr in lrs]
    axes[0, 0].plot(lrs, lr_means, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Learning Rate (α)')
    axes[0, 0].set_ylabel('Avg Final Reward')
    axes[0, 0].set_title('Learning Rate Sensitivity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Gamma Sensitivity
    gammas = sorted(gamma_performance.keys())
    gamma_means = [np.mean(gamma_performance[g]) for g in gammas]
    axes[0, 1].plot(gammas, gamma_means, 's-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Discount Factor (γ)')
    axes[0, 1].set_ylabel('Avg Final Reward')
    axes[0, 1].set_title('Discount Factor Sensitivity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay Sensitivity
    eps_vals = sorted(epsilon_performance.keys())
    eps_means = [np.mean(epsilon_performance[e]) for e in eps_vals]
    axes[1, 0].plot(eps_vals, eps_means, '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Epsilon Decay')
    axes[1, 0].set_ylabel('Avg Final Reward')
    axes[1, 0].set_title('Epsilon Decay Sensitivity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary
    axes[1, 1].axis('off')
    summary_text = "Hyperparameter Tuning Summary:\n\n"
    summary_text += f"Best LR: {lrs[np.argmax(lr_means)]}\n"
    summary_text += f"Best γ: {gammas[np.argmax(gamma_means)]}\n"
    summary_text += f"Best ε_decay: {eps_vals[np.argmax(eps_means)]}\n"
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "hyperparameter_sensitivity.png", dpi=150)
    print("  ✓ Saved: hyperparameter_sensitivity.png")
    
    # ===== PLOT 2: Scenario Comparison =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Across 3 Test Scenarios', fontsize=16, fontweight='bold')
    
    scenarios = list(next(iter(results.values())).keys())
    scenario_rewards = {s: [] for s in scenarios}
    scenario_deliveries = {s: [] for s in scenarios}
    
    for config_data in results.values():
        for scenario in scenarios:
            scenario_rewards[scenario].append(config_data[scenario]["avg_final_reward"])
            scenario_deliveries[scenario].append(config_data[scenario]["avg_final_deliveries"])
    
    # Rewards box plot
    axes[0].boxplot([scenario_rewards[s] for s in scenarios], labels=scenarios)
    axes[0].set_ylabel('Final Reward')
    axes[0].set_title('Reward Distribution Across Scenarios')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Deliveries box plot
    axes[1].boxplot([scenario_deliveries[s] for s in scenarios], labels=scenarios)
    axes[1].set_ylabel('Avg Deliveries (out of 5)')
    axes[1].set_title('Delivery Rate Across Scenarios')
    axes[1].set_ylim([0, 5.5])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "scenario_comparison.png", dpi=150)
    print("  ✓ Saved: scenario_comparison.png")
    
    # ===== PLOT 3: Best Configuration Results Table =====
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    table_data = []
    for scenario, config_info in best_configs.items():
        table_data.append([
            scenario,
            config_info["config"],
            f"{config_info['reward']:.2f}",
            f"{config_info['data']['avg_final_deliveries']:.2f}±{config_info['data']['std_final_deliveries']:.2f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Scenario', 'Best Config', 'Avg Reward', 'Deliveries'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Best Hyperparameter Configurations for Each Scenario', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(plots_dir / "best_configurations_table.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: best_configurations_table.png")
    
    print(f"\n✓ All plots saved to: {plots_dir}/")


if __name__ == "__main__":
    # Load previous results
    results_file = Path("results/tuning/tuning_results.json")
    best_file = Path("results/tuning/best_configurations.json")
    
    if results_file.exists() and best_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        with open(best_file) as f:
            best_configs = json.load(f)
        
        create_comprehensive_plots(results, best_configs)
    else:
        print("❌ Results files not found. Run training first!")
