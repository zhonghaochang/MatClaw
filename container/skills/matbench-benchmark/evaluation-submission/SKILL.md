# Evaluate MatBench Predictions and Submit Results

## When to Use This Skill

This skill should be used when:
- You have a trained model and need to record predictions in the official matbench format
- You want to compute per-fold and aggregate metrics (MAE, RMSE, R2, ROC-AUC, F1)
- You need to compare your results against the leaderboard SOTA scores
- You want publication-quality visualizations (parity plots, residuals, radar charts)
- You are ready to prepare an official submission package for the MatBench leaderboard

## Method Selection

```
What evaluation step do you need?

Recording predictions?
  → Script 1: Record predictions using official protocol

Computing metrics?
  → Script 2: Compute and display per-fold and aggregate metrics

Comparing to leaderboard?
  → Script 3: Leaderboard comparison bar chart visualization

Publication figures?
  → Script 4: Full visualization suite (parity, residual, box, radar)

Ready to submit?
  → Script 5: Prepare official submission directory
```

## Prerequisites

- MatBench conda environment with: matbench, scikit-learn, matplotlib, numpy, pandas
- A trained model that can generate predictions on test data
- All predictions must follow the 5-fold nested cross-validation protocol

## Script 1: Record Predictions (Official Protocol)

This is the correct and complete way to record predictions using the matbench API. Predictions must match the exact length and order of test inputs for each fold.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Record model predictions using the official MatBench protocol.
This is the foundation for all evaluation and submission.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import torch
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure
TASK_NAME = "matbench_mp_e_form"  # Change to your target task
MODEL_NAME = "CGCNN"              # Change to your model name

mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

for task in mb.tasks:
    task.load()
    print(f"\nTask: {task.dataset_name}")
    print(f"Total samples: {len(task.df)}")

    for fold in task.folds:
        print(f"\n  --- Fold {fold} ---")

        # Get training data
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        print(f"  Train samples: {len(train_inputs)}")

        # Get test data (no targets!)
        test_inputs = task.get_test_data(fold, include_target=False)
        print(f"  Test samples:  {len(test_inputs)}")

        # ========================================
        # YOUR MODEL TRAINING AND PREDICTION HERE
        # ========================================
        # Example: load a saved model and predict
        # model = load_model(f"/workspace/group/matbench/models/{MODEL_NAME}/{TASK_NAME}/fold_{fold}.pt")
        # predictions = model.predict(test_inputs)

        # Placeholder -- replace with actual predictions
        predictions = np.zeros(len(test_inputs))
        # ========================================

        # CRITICAL: predictions must be same length as test_inputs
        assert len(predictions) == len(test_inputs), (
            f"Prediction length {len(predictions)} != test length {len(test_inputs)}"
        )

        # Record predictions with model parameters
        task.record(fold, predictions, params={
            "model": MODEL_NAME,
            "lr": 1e-3,
            "epochs": 300,
            # Add all relevant hyperparameters here
        })
        print(f"  Fold {fold} recorded successfully.")

# Validate all folds recorded correctly
mb.validate()
print(f"\n{'='*60}")
print("Validation passed!")
print(f"\nScores:\n{mb.scores}")

# Save results
results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
output_path = f"{results_dir}/{MODEL_NAME}_{TASK_NAME}_results.json.gz"
mb.to_file(output_path)
print(f"\nResults saved to {output_path}")
```

## Script 2: Compute and Display Metrics

Computes per-fold MAE, RMSE, R2 for regression tasks, and ROC-AUC, F1, accuracy for classification tasks. Displays aggregate mean and standard deviation.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Compute detailed per-fold and aggregate metrics from recorded MatBench results.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, accuracy_score
)

# Load saved results
RESULTS_FILE = "/workspace/group/matbench/results/results.json.gz"
mb = MatbenchBenchmark.from_file(RESULTS_FILE)

# Classification tasks
CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}

all_metrics = []

for task in mb.tasks:
    task.load()
    task_name = task.dataset_name
    is_classification = task_name in CLASSIFICATION_TASKS
    print(f"\n{'='*60}")
    print(f"Task: {task_name} ({'Classification' if is_classification else 'Regression'})")
    print(f"{'='*60}")

    fold_metrics = []

    for fold in task.folds:
        # Get true values and predictions
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=True)
        true_values = test_inputs.iloc[:, -1].values if hasattr(test_inputs, 'iloc') else test_inputs

        # Retrieve recorded predictions
        # Access predictions from the task's recorded results
        test_data_with_target = task.get_test_data(fold, include_target=True)
        true_vals = test_data_with_target.values if hasattr(test_data_with_target, 'values') else np.array(test_data_with_target)

        # Get predictions from the benchmark scores
        preds = task.results[fold]

        if is_classification:
            # Classification metrics
            preds_binary = (np.array(preds) >= 0.5).astype(int)
            true_binary = np.array(true_vals).astype(int)

            auc = roc_auc_score(true_binary, preds)
            f1 = f1_score(true_binary, preds_binary)
            acc = accuracy_score(true_binary, preds_binary)

            metrics = {"fold": fold, "ROC-AUC": auc, "F1": f1, "Accuracy": acc}
            print(f"  Fold {fold}: ROC-AUC={auc:.4f}, F1={f1:.4f}, Accuracy={acc:.4f}")
        else:
            # Regression metrics
            true_arr = np.array(true_vals, dtype=float)
            pred_arr = np.array(preds, dtype=float)

            mae = mean_absolute_error(true_arr, pred_arr)
            rmse = np.sqrt(mean_squared_error(true_arr, pred_arr))
            r2 = r2_score(true_arr, pred_arr)

            metrics = {"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2}
            print(f"  Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        fold_metrics.append(metrics)

    # Aggregate
    df_metrics = pd.DataFrame(fold_metrics)
    print(f"\n  Aggregate (mean +/- std):")
    for col in df_metrics.columns:
        if col == "fold":
            continue
        mean_val = df_metrics[col].mean()
        std_val = df_metrics[col].std()
        print(f"    {col}: {mean_val:.4f} +/- {std_val:.4f}")

    all_metrics.append({
        "task": task_name,
        "type": "classification" if is_classification else "regression",
        "fold_metrics": fold_metrics,
        "aggregate": {col: {"mean": df_metrics[col].mean(), "std": df_metrics[col].std()}
                      for col in df_metrics.columns if col != "fold"}
    })

# Save metrics summary
import json
metrics_path = "/workspace/group/matbench/results/metrics_summary.json"
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2, default=float)
print(f"\nMetrics summary saved to {metrics_path}")
```

## Script 3: Leaderboard Comparison Visualization

Bar chart comparing your model's scores against SOTA for each task.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Compare your model results against SOTA leaderboard scores.
Generates a grouped bar chart for visual comparison.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matbench.bench import MatbenchBenchmark

# SOTA reference scores (from sota-reproduction skill)
MATBENCH_SOTA = {
    "matbench_steels":        {"model": "TPOT-Mat",            "score": 79.95,  "metric": "MAE"},
    "matbench_jdft2d":        {"model": "MODNet v0.1.12",      "score": 33.19,  "metric": "MAE"},
    "matbench_phonons":       {"model": "MegNet/kgcnn v2.1.0", "score": 28.76,  "metric": "MAE"},
    "matbench_expt_gap":      {"model": "Darwin",              "score": 0.2865, "metric": "MAE"},
    "matbench_dielectric":    {"model": "MODNet v0.1.12",      "score": 0.2711, "metric": "MAE"},
    "matbench_expt_is_metal": {"model": "Darwin",              "score": 0.9598, "metric": "ROC-AUC"},
    "matbench_glass":         {"model": "MODNet v0.1.12",      "score": 0.9603, "metric": "ROC-AUC"},
    "matbench_log_gvrh":      {"model": "coNGN",               "score": 0.0670, "metric": "MAE"},
    "matbench_log_kvrh":      {"model": "coNGN",               "score": 0.0491, "metric": "MAE"},
    "matbench_perovskites":   {"model": "coGN",                "score": 0.0269, "metric": "MAE"},
    "matbench_mp_gap":        {"model": "coGN",                "score": 0.1559, "metric": "MAE"},
    "matbench_mp_is_metal":   {"model": "CGCNN v2019",         "score": 0.9520, "metric": "ROC-AUC"},
    "matbench_mp_e_form":     {"model": "coGN",                "score": 0.0170, "metric": "MAE"},
}

# Load your results
RESULTS_FILE = "/workspace/group/matbench/results/results.json.gz"
MODEL_NAME = "Your Model"  # Change to your model name

mb = MatbenchBenchmark.from_file(RESULTS_FILE)

# Collect scores
tasks = []
your_scores = []
sota_scores = []
sota_models = []

for task in mb.tasks:
    task_name = task.dataset_name
    if task_name not in MATBENCH_SOTA:
        continue

    # Extract your score from mb.scores
    task_scores = mb.scores.get(task_name, {})
    if not task_scores:
        continue

    # Get the mean score across folds
    your_score = np.mean(list(task_scores.values())) if isinstance(task_scores, dict) else task_scores

    tasks.append(task_name.replace("matbench_", ""))
    your_scores.append(your_score)
    sota_scores.append(MATBENCH_SOTA[task_name]["score"])
    sota_models.append(MATBENCH_SOTA[task_name]["model"])

if not tasks:
    print("No matching tasks found in results. Check your results file.")
    exit(0)

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 1.5), 6))

x = np.arange(len(tasks))
width = 0.35

bars_yours = ax.bar(x - width/2, your_scores, width, label=MODEL_NAME, color="#4C72B0", alpha=0.9)
bars_sota = ax.bar(x + width/2, sota_scores, width, label="SOTA", color="#DD8452", alpha=0.9)

ax.set_xlabel("Task", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"MatBench: {MODEL_NAME} vs SOTA", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars_yours:
    height = bar.get_height()
    ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
for bar in bars_sota:
    height = bar.get_height()
    ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

plt.tight_layout()

plot_dir = "/workspace/group/matbench/plots"
os.makedirs(plot_dir, exist_ok=True)
output_path = f"{plot_dir}/leaderboard_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Leaderboard comparison saved to {output_path}")

# Also print a text table
print(f"\n{'Task':<25s} {'Yours':>10s} {'SOTA':>10s} {'SOTA Model':<25s} {'Gap':>10s}")
print("-" * 85)
for i, t in enumerate(tasks):
    gap = your_scores[i] - sota_scores[i]
    sign = "+" if gap > 0 else ""
    print(f"{t:<25s} {your_scores[i]:>10.4f} {sota_scores[i]:>10.4f} {sota_models[i]:<25s} {sign}{gap:>9.4f}")
```

## Script 4: Publication Visualizations

Complete plotting suite for publication-quality figures: parity plot, residual histogram, per-fold box plot, and multi-task radar chart.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Publication-quality visualization suite for MatBench results.
Generates: parity plot, residual histogram, per-fold box plot, radar chart.
All saved to /workspace/group/matbench/plots/
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matbench.bench import MatbenchBenchmark
from sklearn.metrics import mean_absolute_error, r2_score

plot_dir = "/workspace/group/matbench/plots"
os.makedirs(plot_dir, exist_ok=True)

RESULTS_FILE = "/workspace/group/matbench/results/results.json.gz"
mb = MatbenchBenchmark.from_file(RESULTS_FILE)

CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}

for task in mb.tasks:
    task.load()
    task_name = task.dataset_name
    short_name = task_name.replace("matbench_", "")
    is_classification = task_name in CLASSIFICATION_TASKS

    print(f"\nGenerating plots for {task_name}...")

    all_true = []
    all_pred = []
    fold_errors = {}

    for fold in task.folds:
        test_data = task.get_test_data(fold, include_target=True)
        true_vals = np.array(test_data.values if hasattr(test_data, 'values') else test_data, dtype=float)
        preds = np.array(task.results[fold], dtype=float)

        all_true.extend(true_vals)
        all_pred.extend(preds)
        fold_errors[fold] = np.abs(true_vals - preds)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    if is_classification:
        print(f"  Skipping detailed plots for classification task {task_name}")
        continue

    # --- Plot 1: Parity Plot with Density Coloring ---
    fig, ax = plt.subplots(figsize=(7, 7))

    from scipy.stats import gaussian_kde
    xy = np.vstack([all_true, all_pred])
    try:
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        x_sorted, y_sorted, z_sorted = all_true[idx], all_pred[idx], density[idx]
        scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=5, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Density")
    except Exception:
        ax.scatter(all_true, all_pred, s=5, alpha=0.3, color="#4C72B0")

    # Perfect prediction line
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            "r--", lw=1.5, alpha=0.7, label="y = x")

    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)
    ax.set_xlabel("True Value", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(f"{short_name}: Parity Plot\nMAE={mae:.4f}, R2={r2:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{short_name}_parity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Parity plot saved: {plot_dir}/{short_name}_parity.png")

    # --- Plot 2: Residual Histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = all_pred - all_true
    ax.hist(residuals, bins=100, color="#4C72B0", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=residuals.mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean={residuals.mean():.4f}")
    ax.set_xlabel("Residual (Predicted - True)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{short_name}: Residual Distribution", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{short_name}_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Residual histogram saved: {plot_dir}/{short_name}_residuals.png")

    # --- Plot 3: Per-Fold Box Plot of Errors ---
    fig, ax = plt.subplots(figsize=(8, 5))
    fold_labels = sorted(fold_errors.keys())
    box_data = [fold_errors[f] for f in fold_labels]
    bp = ax.boxplot(box_data, labels=[f"Fold {f}" for f in fold_labels],
                    patch_artist=True, showfliers=False)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title(f"{short_name}: Per-Fold Error Distribution", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{short_name}_fold_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Box plot saved: {plot_dir}/{short_name}_fold_boxplot.png")

# --- Plot 4: Multi-Task Radar Chart (Normalized Scores) ---
# Only generated when multiple tasks are present

MATBENCH_SOTA = {
    "matbench_steels": 79.95, "matbench_jdft2d": 33.19, "matbench_phonons": 28.76,
    "matbench_expt_gap": 0.2865, "matbench_dielectric": 0.2711,
    "matbench_expt_is_metal": 0.9598, "matbench_glass": 0.9603,
    "matbench_log_gvrh": 0.0670, "matbench_log_kvrh": 0.0491,
    "matbench_perovskites": 0.0269, "matbench_mp_gap": 0.1559,
    "matbench_mp_is_metal": 0.9520, "matbench_mp_e_form": 0.0170,
}

task_names = []
normalized_scores = []

for task in mb.tasks:
    task_name = task.dataset_name
    if task_name not in MATBENCH_SOTA:
        continue
    task_scores = mb.scores.get(task_name, {})
    if not task_scores:
        continue
    your_score = np.mean(list(task_scores.values())) if isinstance(task_scores, dict) else task_scores
    sota_score = MATBENCH_SOTA[task_name]

    # Normalize: for MAE tasks, ratio = sota/yours (lower is better, so higher ratio = better)
    # For AUC tasks, ratio = yours/sota (higher is better)
    if task_name in CLASSIFICATION_TASKS:
        ratio = your_score / sota_score if sota_score != 0 else 0
    else:
        ratio = sota_score / your_score if your_score != 0 else 0

    task_names.append(task_name.replace("matbench_", ""))
    normalized_scores.append(min(ratio, 2.0))  # Cap at 2.0 for display

if len(task_names) >= 3:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(task_names), endpoint=False).tolist()
    normalized_scores_plot = normalized_scores + [normalized_scores[0]]
    angles_plot = angles + [angles[0]]

    ax.fill(angles_plot, normalized_scores_plot, color="#4C72B0", alpha=0.25)
    ax.plot(angles_plot, normalized_scores_plot, color="#4C72B0", linewidth=2)
    ax.scatter(angles, normalized_scores, color="#4C72B0", s=50, zorder=5)

    # SOTA reference (circle at 1.0)
    sota_ref = [1.0] * (len(task_names) + 1)
    ax.plot(angles_plot, sota_ref, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="SOTA")

    ax.set_xticks(angles)
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_ylim(0, max(max(normalized_scores) * 1.1, 1.5))
    ax.set_title("Normalized Performance vs SOTA\n(1.0 = matches SOTA)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nRadar chart saved: {plot_dir}/radar_chart.png")

print(f"\nAll plots saved to {plot_dir}/")
```

## Script 5: Prepare Official Submission

Creates the complete submission directory with all required files for the MatBench leaderboard.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Prepare an official MatBench leaderboard submission.
Creates the complete submission directory with all required files.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import json
import shutil
from datetime import datetime
from matbench.bench import MatbenchBenchmark

# Configuration -- update these for your submission
ALGORITHM_NAME = "Your Model Name"
AUTHORS = ["Author One", "Author Two"]
ALGORITHM_DESCRIPTION = (
    "Brief description of your model architecture and approach. "
    "Include key details: model type, featurization, training strategy."
)
BIBTEX = """
@article{yourmodel2025,
  title={Your Model: A Crystal Property Prediction Model},
  author={One, Author and Two, Author},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
"""
NOTES = "Trained on NVIDIA A100-SXM4-80GB. 5-fold nested CV with official matbench splits."
REQUIREMENTS = {
    "python": "3.11",
    "matbench": "0.6",
    "torch": "2.x",
    "torch_geometric": "2.x",
}

# Source results file
RESULTS_FILE = "/workspace/group/matbench/results/results.json.gz"

# Submission directory
submission_dir = "/workspace/group/matbench/submission"
os.makedirs(submission_dir, exist_ok=True)

# Step 1: Copy results.json.gz
print("Step 1: Copying results file...")
results_dest = os.path.join(submission_dir, "results.json.gz")
shutil.copy2(RESULTS_FILE, results_dest)
print(f"  Copied to {results_dest}")

# Validate the results
mb = MatbenchBenchmark.from_file(results_dest)
mb.validate()
print("  Results validation passed.")
print(f"  Tasks: {[t.dataset_name for t in mb.tasks]}")
print(f"  Scores: {mb.scores}")

# Step 2: Create info.json
print("\nStep 2: Creating info.json...")
info = {
    "authors": AUTHORS,
    "algorithm": ALGORITHM_NAME,
    "algorithm_long": ALGORITHM_DESCRIPTION,
    "bibtex_refs": [BIBTEX.strip()],
    "notes": NOTES,
    "requirements": REQUIREMENTS,
    "date_created": datetime.now().isoformat(),
    "matbench_version": "0.6",
    "tasks": {},
}

# Add per-task scores and params
for task in mb.tasks:
    task_name = task.dataset_name
    task_scores = mb.scores.get(task_name, {})
    info["tasks"][task_name] = {
        "scores": task_scores,
        "params": task.params if hasattr(task, 'params') else {},
    }

info_path = os.path.join(submission_dir, "info.json")
with open(info_path, "w") as f:
    json.dump(info, f, indent=2, default=str)
print(f"  Created {info_path}")

# Step 3: Create benchmark_code.py template
print("\nStep 3: Creating benchmark_code.py...")
benchmark_code = f'''#!/opt/conda/envs/matbench/bin/python
"""
{ALGORITHM_NAME} - MatBench Benchmark Code
Authors: {", ".join(AUTHORS)}
Date: {datetime.now().strftime("%Y-%m-%d")}

This script reproduces the benchmark results for {ALGORITHM_NAME}.
Run with: /opt/conda/envs/matbench/bin/python benchmark_code.py
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import torch
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {{device}}")

# TODO: Import your model here
# from your_model import YourModel

# Run full benchmark
mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    print(f"\\nTask: {{task.dataset_name}}")

    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=False)

        # TODO: Train model and predict
        # model = YourModel()
        # model.fit(train_inputs, train_outputs)
        # predictions = model.predict(test_inputs)

        predictions = np.zeros(len(test_inputs))  # Replace
        task.record(fold, predictions)
        print(f"  Fold {{fold}} done")

mb.validate()
print(f"\\nFinal scores:\\n{{mb.scores}}")

results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
mb.to_file(f"{{results_dir}}/results.json.gz")
'''

benchmark_path = os.path.join(submission_dir, "benchmark_code.py")
with open(benchmark_path, "w") as f:
    f.write(benchmark_code)
print(f"  Created {benchmark_path}")

# Summary
print(f"\n{'='*60}")
print("Submission directory ready!")
print(f"{'='*60}")
print(f"\nDirectory: {submission_dir}/")
print(f"  results.json.gz  -- benchmark results ({os.path.getsize(results_dest)} bytes)")
print(f"  info.json         -- metadata and configuration")
print(f"  benchmark_code.py -- reproducible training script")
print(f"\nNext steps:")
print(f"  1. Edit benchmark_code.py with your actual model code")
print(f"  2. Update info.json with accurate algorithm description and bibtex")
print(f"  3. Follow matbench submission instructions at:")
print(f"     https://matbench.materialsproject.org/How%20To%20Submit/")
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| n_folds | Number of CV folds (fixed by matbench) | 5 |
| metric (regression) | MAE (primary), RMSE, R2 | MAE |
| metric (classification) | ROC-AUC (primary), F1, Accuracy | ROC-AUC |
| output_format | results.json.gz (gzipped JSON) | json.gz |
| plot_dpi | Resolution for saved figures | 150 |

## Common Issues

| Issue | Solution |
|-------|---------|
| Prediction length mismatch | Ensure `len(predictions) == len(test_inputs)` for each fold; this is the most common error |
| "Fold already recorded" error | Each fold can only be recorded once per task; create a new MatbenchBenchmark instance to re-record |
| Classification predictions as float vs int | matbench expects float probabilities for classification (not binary 0/1); ROC-AUC needs continuous scores |
| NumPy serialization error on save | Convert predictions to plain Python floats: `predictions = [float(x) for x in predictions]` |
| Metric differences from leaderboard | Verify you are using the exact matbench fold splits and not shuffling data; use `autoload=False` and `task.load()` |
| mb.scores shows empty dict | Ensure all folds are recorded and `mb.validate()` passes before accessing scores |
| Large results file | results.json.gz is compressed; typical size is <10MB even for all 13 tasks |
