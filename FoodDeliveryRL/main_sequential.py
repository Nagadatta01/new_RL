"""
MAIN ORCHESTRATOR - Sequential Execution
Step 1 → 2 → 3A (Baseline) → 3B (Tuning) → 4 → 5

NO console input waits - runs automatically!
"""

print("\n" + "="*80)
print("DQN PROJECT - SEQUENTIAL EXECUTION")
print("="*80)

# STEP 1
print("\n\n🟢 STEP 1: BASELINE SETUP...")
from step1_baseline_setup import step1_baseline_setup
config = step1_baseline_setup()
print("✓ Step 1 complete!")

# STEP 2
print("\n\n🟡 STEP 2: HYPERPARAMETER RATIONALE...")
from step2_hyperparameter_rationale import step2_hyperparameter_rationale
step2_hyperparameter_rationale()
print("✓ Step 2 complete!")

# STEP 3A: Baseline Training
print("\n\n🟠 STEP 3A: BASELINE TRAINING...")
from step3_baseline_training_only import step3_baseline_training
baseline_results = step3_baseline_training()
print("✓ Step 3A complete!")

# STEP 3B: Ablation Study
print("\n\n🔴 STEP 3B: HYPERPARAMETER TUNING (Ablation Study)...")
from step3_ablation_study_tuning import step3_ablation_study
ablation_results, best_config = step3_ablation_study()
print("✓ Step 3B complete!")

# STEP 4
print("\n\n🔵 STEP 4: TEST & EVALUATION...")
from step4_test_evaluation import step4_test_evaluation
step4_test_evaluation()
print("✓ Step 4 complete!")

# STEP 5
print("\n\n🟣 STEP 5: DISCUSSION & CONCLUSION...")
from step5_discussion_and_conclusion import step5_discussion_conclusion
step5_discussion_conclusion()
print("✓ Step 5 complete!")

print("\n" + "="*80)
print("✅ PROJECT COMPLETE!")
print("="*80)
print("\n📁 Results saved in:")
print("  • results/step1_baseline/")
print("  • results/step2_rationale/")
print("  • results/step3_baseline/     ← BASELINE RESULTS")
print("  • results/step3_ablation/     ← TUNING RESULTS")
print("  • results/step4_evaluation/   ← TEST RESULTS")
print("  • results/step5_discussion/   ← FINAL ANALYSIS")
print("\n")
