---
name: task-setup
description: Load MatBench tasks, explore data, visualize distributions, and understand the 5-fold nested cross-validation protocol
---

# Load MatBench Tasks and Explore Data

## When to Use

- Starting a MatBench benchmark and need to understand the data landscape
- Loading a specific task and inspecting fold structure
- Understanding the 5-fold nested cross-validation protocol
- Visualizing target distributions, structure statistics, and composition space
- Checking dataset statistics (size, class balance, outliers)

## Method Selection

```
What do you need?
│
├─ Overview of all 13 tasks?
│  └─ Script 1: Multi-task overview
│
├─ Deep dive into one task?
│  └─ Script 2: Single task deep dive
│     ├─ Composition input? → Good for composition-models next
│     └─ Structure input? → Good for structure-gnn next
│
├─ Visualize distributions?
│  └─ Script 3: Data visualization
│     ├─ Regression task? → Histogram + box plot
│     └─ Classification task? → Class balance bar chart
│
└─ Just want to run something end-to-end quickly?
   └─ Script 4: Quick-start template (dummy model)
```

## Prerequisites

- Python: `/opt/conda/envs/matbench/bin/python`
- Packages: matbench, pymatgen, pandas, numpy, matplotlib, matminer

## Script 1: Multi-Task Overview

Load all 13 MatBench tasks and print a comprehensive summary table.

```python
#!/opt/conda/envs/matbench/bin/python
"""Load all 13 MatBench tasks and print a summary overview table."""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

print("=" * 80)
print("MatBench: Loading all 13 tasks...")
print("=" * 80)

mb = MatbenchBenchmark(autoload=False)

TASK_META = {
    "matbench_steels": ("Composition", "Regression", "Yield strength (MPa)"),
    "matbench_jdft2d": ("Structure", "Regression", "Exfoliation energy (meV/atom)"),
    "matbench_phonons": ("Structure", "Regression", "Max phonon freq (1/cm)"),
    "matbench_expt_gap": ("Composition", "Regression", "Band gap (eV)"),
    "matbench_dielectric": ("Composition", "Regression", "Dielectric constant"),
    "matbench_expt_is_metal": ("Composition", "Classification", "Metal / non-metal"),
    "matbench_glass": ("Composition", "Classification", "Glass-forming ability"),
    "matbench_log_gvrh": ("Structure", "Regression", "log10 bulk modulus (GPa)"),
    "matbench_log_kvrh": ("Structure", "Regression", "log10 shear modulus (GPa)"),
    "matbench_perovskites": ("Structure", "Regression", "Formation energy (eV)"),
    "matbench_mp_gap": ("Structure", "Regression", "Band gap (eV)"),
    "matbench_mp_is_metal": ("Structure", "Classification", "Metal / non-metal"),
    "matbench_mp_e_form": ("Structure", "Regression", "Formation energy (eV/atom)"),
}

rows = []
for task in mb.tasks:
    task_name = task.dataset_name
    print(f"  Loading {task_name}...")
    task.load()
    df = task.df

    input_type, output_type, prop = TASK_META[task_name]
    target_col = df.columns[-1]
    targets = df[target_col]

    row = {
        "Task": task_name,
        "Input": input_type,
        "Type": output_type,
        "Samples": len(df),
        "Property": prop,
    }

    if output_type == "Regression":
        row["Mean"] = f"{targets.mean():.4f}"
        row["Std"] = f"{targets.std():.4f}"
        row["Min"] = f"{targets.min():.4f}"
        row["Max"] = f"{targets.max():.4f}"
    else:
        counts = targets.value_counts()
        row["Mean"] = f"Class0={counts.iloc[0]}"
        row["Std"] = f"Class1={counts.iloc[1]}"
        row["Min"] = f"Ratio={counts.iloc[1]/counts.iloc[0]:.2f}"
        row["Max"] = "N/A"

    rows.append(row)

summary_df = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("MatBench Tasks Summary")
print("=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)

# Save summary to CSV
out_path = "/workspace/group/matbench/results/task_summary.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
summary_df.to_csv(out_path, index=False)
print(f"\nSummary saved to {out_path}")
print("Done!")
```

## Script 2: Single Task Deep Dive

Load one task, inspect fold structure, and explore samples.

```python
#!/opt/conda/envs/matbench/bin/python
"""Deep dive into a single MatBench task: fold structure, sample inspection."""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

# === CONFIGURE THIS ===
TASK_NAME = "matbench_mp_e_form"  # Change to any of the 13 tasks
# ======================

print(f"Loading task: {TASK_NAME}")
mb = MatbenchBenchmark(autoload=False)

task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break

if task is None:
    raise ValueError(f"Task {TASK_NAME} not found")

task.load()
print(f"Task loaded: {task.dataset_name}")
print(f"Total samples: {len(task.df)}")
print(f"Columns: {list(task.df.columns)}")
print()

# Inspect the full dataset
df = task.df
target_col = df.columns[-1]
input_col = df.columns[0]

print(f"Input column: '{input_col}' (type: {type(df[input_col].iloc[0]).__name__})")
print(f"Target column: '{target_col}' (dtype: {df[target_col].dtype})")
print()

# Target statistics
targets = df[target_col]
print("=== Target Statistics ===")
print(f"  Count:  {len(targets)}")
print(f"  Mean:   {targets.mean():.6f}")
print(f"  Std:    {targets.std():.6f}")
print(f"  Min:    {targets.min():.6f}")
print(f"  Max:    {targets.max():.6f}")
print(f"  Median: {targets.median():.6f}")
print(f"  25%:    {targets.quantile(0.25):.6f}")
print(f"  75%:    {targets.quantile(0.75):.6f}")
print()

# Inspect folds
print("=== 5-Fold Cross-Validation Structure ===")
for fold_idx in range(5):
    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Fold {fold_idx}: train={len(train_inputs)}, test={len(test_inputs)}")
print()

# Show first few samples
print("=== First 5 Samples ===")
for i in range(min(5, len(df))):
    inp = df[input_col].iloc[i]
    tgt = df[target_col].iloc[i]
    inp_str = str(inp)[:80] + "..." if len(str(inp)) > 80 else str(inp)
    print(f"  [{i}] Input: {inp_str}")
    print(f"       Target: {tgt}")
print()

print("Done!")
```

## Script 3: Data Visualization

Generate histograms, box plots, and structural statistics for a given task.

```python
#!/opt/conda/envs/matbench/bin/python
"""Visualize MatBench task data: target distributions, structure stats."""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)
os.makedirs("/workspace/group/matbench/plots", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

# === CONFIGURE THIS ===
TASK_NAME = "matbench_mp_e_form"  # Change to any of the 13 tasks
# ======================

print(f"Loading task: {TASK_NAME}")
mb = MatbenchBenchmark(autoload=False)
task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break
task.load()

df = task.df
target_col = df.columns[-1]
input_col = df.columns[0]
targets = df[target_col]

plot_dir = f"/workspace/group/matbench/plots/{TASK_NAME}"
os.makedirs(plot_dir, exist_ok=True)

# --- Plot 1: Target distribution histogram ---
print("Plotting target distribution histogram...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(targets, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
ax.set_xlabel(target_col, fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"{TASK_NAME}: Target Distribution", fontsize=14)
ax.axvline(targets.mean(), color="red", linestyle="--", label=f"Mean={targets.mean():.4f}")
ax.axvline(targets.median(), color="orange", linestyle="--", label=f"Median={targets.median():.4f}")
ax.legend(fontsize=11)
plt.tight_layout()
path1 = f"{plot_dir}/target_histogram.png"
fig.savefig(path1, dpi=150)
plt.close(fig)
print(f"  Saved: {path1}")

# --- Plot 2: Box plot per fold ---
print("Plotting per-fold box plots...")
fold_targets = []
fold_labels = []
for fold_idx in range(5):
    _, train_outputs = task.get_train_and_val_data(fold_idx)
    fold_targets.append(train_outputs.values)
    fold_labels.append(f"Fold {fold_idx}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(fold_targets, labels=fold_labels, patch_artist=True,
           boxprops=dict(facecolor="lightblue"))
ax.set_ylabel(target_col, fontsize=12)
ax.set_title(f"{TASK_NAME}: Target Distribution per Fold (Train)", fontsize=14)
plt.tight_layout()
path2 = f"{plot_dir}/fold_boxplot.png"
fig.savefig(path2, dpi=150)
plt.close(fig)
print(f"  Saved: {path2}")

# --- Plot 3: Structure-specific plots (if input is structure) ---
sample_input = df[input_col].iloc[0]
if hasattr(sample_input, "lattice"):
    print("Detected structure input. Plotting structural statistics...")

    # Number of atoms histogram
    num_atoms = df[input_col].apply(lambda s: len(s))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(num_atoms, bins=100, edgecolor="black", alpha=0.7, color="coral")
    ax.set_xlabel("Number of atoms", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{TASK_NAME}: Number of Atoms per Structure", fontsize=14)
    ax.axvline(num_atoms.mean(), color="red", linestyle="--",
               label=f"Mean={num_atoms.mean():.1f}")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path3 = f"{plot_dir}/num_atoms_histogram.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # Space group distribution (top 20)
    print("Computing space group distribution (this may take a moment)...")
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    spacegroups = []
    for i, struct in enumerate(df[input_col]):
        try:
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            spacegroups.append(sga.get_space_group_number())
        except Exception:
            spacegroups.append(-1)
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(df)} structures...")

    sg_series = pd.Series(spacegroups)
    top20 = sg_series.value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    top20.plot(kind="bar", ax=ax, color="teal", edgecolor="black")
    ax.set_xlabel("Space Group Number", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{TASK_NAME}: Top 20 Space Groups", fontsize=14)
    plt.tight_layout()
    path4 = f"{plot_dir}/spacegroup_distribution.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path4}")
else:
    print("Input is composition-based. Skipping structure-specific plots.")

print(f"\nAll plots saved to {plot_dir}/")
print("Done!")
```

## Script 4: Quick-Start Template (Dummy Model)

Minimal complete example showing the correct MatBench API workflow: load, predict, record, save.

```python
#!/opt/conda/envs/matbench/bin/python
"""
MatBench Quick-Start Template: Dummy model (mean predictor).
This is the minimal complete workflow showing how to use the matbench API.
Copy this and replace the model section with your own model.
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)
os.makedirs("/workspace/group/matbench/results", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from matbench.bench import MatbenchBenchmark

# === CONFIGURE THIS ===
TASK_NAME = "matbench_steels"  # Change to any of the 13 tasks
MODEL_NAME = "DummyMeanPredictor"
# ======================

print(f"=" * 60)
print(f"MatBench Quick-Start: {MODEL_NAME} on {TASK_NAME}")
print(f"=" * 60)

# Step 1: Create benchmark and get the task
print("\nStep 1: Loading benchmark and task...")
mb = MatbenchBenchmark(autoload=False)
task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break
task.load()
print(f"  Task loaded: {task.dataset_name} ({len(task.df)} samples)")

# Step 2: Run 5-fold cross-validation
print("\nStep 2: Running 5-fold cross-validation...")
fold_scores = []

for fold_idx in range(5):
    print(f"\n  --- Fold {fold_idx} ---")

    # Get training data
    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    print(f"  Train samples: {len(train_inputs)}")

    # Get test data (no targets)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Test samples:  {len(test_inputs)}")

    # ============================================================
    # YOUR MODEL HERE: Replace this section with your own model
    # ============================================================
    # Dummy model: predict the training set mean for all test samples
    train_mean = train_outputs.mean()
    predictions = np.full(len(test_inputs), train_mean)
    print(f"  Prediction: constant = {train_mean:.4f}")
    # ============================================================

    # Step 3: Record predictions (MUST match test_inputs index and length)
    task.record(fold_idx, predictions)

    # Calculate fold score (for logging only; matbench computes official scores)
    test_targets = task.get_test_data(fold_idx, include_target=True)
    target_col = test_targets.columns[-1]
    true_values = test_targets[target_col].values
    mae = np.mean(np.abs(true_values - predictions))
    fold_scores.append(mae)
    print(f"  Fold {fold_idx} MAE: {mae:.4f}")

# Step 4: Print summary
print(f"\n{'=' * 60}")
print(f"Results Summary: {MODEL_NAME} on {TASK_NAME}")
print(f"{'=' * 60}")
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: MAE = {score:.4f}")
print(f"  Mean MAE: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Step 5: Save results
results_path = f"/workspace/group/matbench/results/{TASK_NAME}_{MODEL_NAME}.json"
results = {
    "task": TASK_NAME,
    "model": MODEL_NAME,
    "fold_scores_mae": fold_scores,
    "mean_mae": float(np.mean(fold_scores)),
    "std_mae": float(np.std(fold_scores)),
}
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print("\nDone! Replace the dummy model section with your own model to get started.")
```

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `task_name` | Name of the MatBench task | `"matbench_mp_e_form"` |
| `fold` | Fold index for cross-validation | `0` to `4` |
| `autoload` | Whether to auto-load all tasks | `False` (recommended: load individually) |
| `MATBENCH_DATA_HOME` | Directory for dataset cache | `/workspace/group/matbench/data` |
| `as_type` | Return input as specific type | `"tuple"` for (input, target) pairs |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'matbench'` | Wrong Python environment | Use `/opt/conda/envs/matbench/bin/python` |
| Download hangs or times out | Network issue or large dataset | Set `MATBENCH_DATA_HOME` to data disk; retry; check proxy |
| `MemoryError` on large tasks | matbench_mp_e_form has 132k structures | Process in batches; use structure fingerprints instead of full graphs |
| Wrong prediction length error | `len(predictions) != len(test_inputs)` | Ensure predictions array matches test set size exactly |
| `ValueError: Fold already recorded` | Called `task.record()` twice for same fold | Create a fresh `MatbenchBenchmark` instance |
| Plots not displaying | Running headless (no display) | Always use `matplotlib.use("Agg")` before importing pyplot |
