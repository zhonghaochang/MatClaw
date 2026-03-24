---
name: composition-models
description: Build composition-based ML models (RF, GBR, MLP + Magpie features) for MatBench tasks with composition string input
---

# Build Composition-Based Models for MatBench

## ⚠️ For PyTorch MLP Models — A100 Optimization

When using the MLP (Script 4), read `~/.claude/skills/matbench-benchmark/training-pipeline/SKILL.md` "A100 GPU Optimization" section first. Use `batch_size=2048+`, `pin_memory=True`, `num_workers=4`, AMP autocast, and TF32.

## When to Use

- Task input is a composition string (matbench_steels, matbench_expt_gap, matbench_expt_is_metal, matbench_glass, matbench_dielectric)
- Want a strong baseline with traditional ML before trying neural approaches
- Need Magpie featurization via matminer's `ElementProperty`
- Want to bridge from sklearn to composition-based neural approaches (MLP)

## Method Selection

```
What is your task?
│
├─ < 1,000 samples? (matbench_steels: 312)
│  └─ RandomForestRegressor + Magpie
│     (robust on small data, less prone to overfitting)
│
├─ 1,000 - 6,000 samples? (expt_gap, dielectric, expt_is_metal, glass)
│  ├─ Regression? → RF or GBR + Magpie (GBR often slightly better)
│  └─ Classification? → RandomForestClassifier + Magpie (use ROC-AUC)
│
├─ Want feature importance analysis?
│  └─ Script 3: RF feature importances + top-20 plot
│
└─ Want a neural baseline on composition features?
   └─ Script 4: PyTorch MLP on Magpie features (GPU-accelerated)
```

**Metric guidance:**
- Regression tasks: MAE (mean absolute error)
- Classification tasks: ROC-AUC

## Prerequisites

- Python: `/opt/conda/envs/matbench/bin/python`
- Packages: matbench, matminer, scikit-learn, pandas, numpy, matplotlib, torch

## Script 1: Random Forest + Magpie (Full Pipeline)

Complete MatBench-compliant benchmark script with Magpie featurization and RandomForest.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Random Forest + Magpie features: Full MatBench-compliant benchmark pipeline.
Works for any composition-input task (regression or classification).
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)
os.makedirs("/workspace/group/matbench/results", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"  # Composition-input task
IS_CLASSIFICATION = False         # True for expt_is_metal, glass
N_ESTIMATORS = 500
N_JOBS = -1
# ======================

print(f"=" * 70)
print(f"Random Forest + Magpie | {TASK_NAME}")
print(f"=" * 70)

# Featurizer setup
featurizer = ElementProperty.from_preset("magpie")

def featurize_compositions(compositions):
    """Convert composition strings to Magpie feature vectors."""
    comp_objects = [Composition(c) if isinstance(c, str) else c for c in compositions]
    df_comp = pd.DataFrame({"composition": comp_objects})
    print(f"    Featurizing {len(comp_objects)} compositions...")
    df_feat = featurizer.featurize_dataframe(df_comp, "composition", ignore_errors=True)
    df_feat = df_feat.drop(columns=["composition"])
    # Handle NaN: impute with 0
    df_feat = df_feat.fillna(0)
    return df_feat

# Load task
print("\nLoading task...")
mb = MatbenchBenchmark(autoload=False)
task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break
task.load()
print(f"  Loaded: {task.dataset_name} ({len(task.df)} samples)")

# 5-fold cross-validation
fold_scores = []

for fold_idx in range(5):
    print(f"\n--- Fold {fold_idx} ---")

    # Get data
    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    # Featurize
    print("  Featurizing train set...")
    X_train = featurize_compositions(train_inputs)
    print("  Featurizing test set...")
    X_test = featurize_compositions(test_inputs)
    y_train = train_outputs.values

    # Train model
    print(f"  Training RandomForest (n_estimators={N_ESTIMATORS})...")
    if IS_CLASSIFICATION:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=42
        )
    model.fit(X_train.values, y_train)

    # Predict
    if IS_CLASSIFICATION:
        predictions = model.predict_proba(X_test.values)[:, 1]
    else:
        predictions = model.predict(X_test.values)

    # Record predictions
    task.record(fold_idx, predictions)

    # Score (for logging)
    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        score = roc_auc_score(true_values, predictions)
        metric = "ROC-AUC"
    else:
        score = mean_absolute_error(true_values, predictions)
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: RF + Magpie on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results_path = f"/workspace/group/matbench/results/{TASK_NAME}_RF_Magpie.json"
results = {
    "task": TASK_NAME,
    "model": "RandomForest_Magpie",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "n_estimators": N_ESTIMATORS,
}
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
print("Done!")
```

## Script 2: Gradient Boosting + Magpie

Same structure as Script 1 but using GradientBoostingRegressor/Classifier for potentially better accuracy.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Gradient Boosting + Magpie features: Full MatBench-compliant benchmark.
Often slightly better than RF on medium-sized composition tasks.
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)
os.makedirs("/workspace/group/matbench/results", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"  # Composition-input task
IS_CLASSIFICATION = False         # True for expt_is_metal, glass
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
MAX_DEPTH = 6
# ======================

print(f"=" * 70)
print(f"Gradient Boosting + Magpie | {TASK_NAME}")
print(f"=" * 70)

# Featurizer setup
featurizer = ElementProperty.from_preset("magpie")

def featurize_compositions(compositions):
    """Convert composition strings to Magpie feature vectors."""
    comp_objects = [Composition(c) if isinstance(c, str) else c for c in compositions]
    df_comp = pd.DataFrame({"composition": comp_objects})
    print(f"    Featurizing {len(comp_objects)} compositions...")
    df_feat = featurizer.featurize_dataframe(df_comp, "composition", ignore_errors=True)
    df_feat = df_feat.drop(columns=["composition"])
    df_feat = df_feat.fillna(0)
    return df_feat

# Load task
print("\nLoading task...")
mb = MatbenchBenchmark(autoload=False)
task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break
task.load()
print(f"  Loaded: {task.dataset_name} ({len(task.df)} samples)")

# 5-fold cross-validation
fold_scores = []

for fold_idx in range(5):
    print(f"\n--- Fold {fold_idx} ---")

    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    print("  Featurizing train set...")
    X_train = featurize_compositions(train_inputs)
    print("  Featurizing test set...")
    X_test = featurize_compositions(test_inputs)
    y_train = train_outputs.values

    print(f"  Training GBR (n_estimators={N_ESTIMATORS}, lr={LEARNING_RATE}, depth={MAX_DEPTH})...")
    if IS_CLASSIFICATION:
        model = GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            random_state=42,
        )
    model.fit(X_train.values, y_train)

    if IS_CLASSIFICATION:
        predictions = model.predict_proba(X_test.values)[:, 1]
    else:
        predictions = model.predict(X_test.values)

    task.record(fold_idx, predictions)

    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        score = roc_auc_score(true_values, predictions)
        metric = "ROC-AUC"
    else:
        score = mean_absolute_error(true_values, predictions)
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: GBR + Magpie on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results_path = f"/workspace/group/matbench/results/{TASK_NAME}_GBR_Magpie.json"
results = {
    "task": TASK_NAME,
    "model": "GradientBoosting_Magpie",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "max_depth": MAX_DEPTH,
}
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
print("Done!")
```

## Script 3: Feature Importance Analysis

Extract and visualize the most important Magpie features from a trained RandomForest model.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Feature Importance Analysis: Train RF on one fold, extract and plot top-20 Magpie features.
Helps understand which elemental properties drive predictions.
"""

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
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"  # Composition-input task
IS_CLASSIFICATION = False
FOLD_IDX = 0                      # Which fold to analyze
N_ESTIMATORS = 500
TOP_N = 20                        # Number of top features to plot
# ======================

print(f"=" * 70)
print(f"Feature Importance Analysis | {TASK_NAME} | Fold {FOLD_IDX}")
print(f"=" * 70)

# Featurizer
featurizer = ElementProperty.from_preset("magpie")

def featurize_compositions(compositions):
    comp_objects = [Composition(c) if isinstance(c, str) else c for c in compositions]
    df_comp = pd.DataFrame({"composition": comp_objects})
    print(f"    Featurizing {len(comp_objects)} compositions...")
    df_feat = featurizer.featurize_dataframe(df_comp, "composition", ignore_errors=True)
    df_feat = df_feat.drop(columns=["composition"])
    df_feat = df_feat.fillna(0)
    return df_feat

# Load task
print("\nLoading task...")
mb = MatbenchBenchmark(autoload=False)
task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break
task.load()
print(f"  Loaded: {task.dataset_name} ({len(task.df)} samples)")

# Get fold data
train_inputs, train_outputs = task.get_train_and_val_data(FOLD_IDX)
print(f"  Train samples: {len(train_inputs)}")

# Featurize
print("  Featurizing...")
X_train = featurize_compositions(train_inputs)
y_train = train_outputs.values
feature_names = X_train.columns.tolist()
print(f"  Number of Magpie features: {len(feature_names)}")

# Train model
print(f"  Training RandomForest (n_estimators={N_ESTIMATORS})...")
if IS_CLASSIFICATION:
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42)
else:
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=42)
model.fit(X_train.values, y_train)

# Extract feature importances
importances = model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances,
}).sort_values("Importance", ascending=False)

# Print top features
print(f"\n  Top {TOP_N} Most Important Features:")
print(f"  {'Feature':<50} {'Importance':>10}")
print(f"  {'-'*60}")
for _, row in feat_imp.head(TOP_N).iterrows():
    print(f"  {row['Feature']:<50} {row['Importance']:>10.4f}")

# Plot
print("\n  Generating plot...")
plot_dir = f"/workspace/group/matbench/plots/{TASK_NAME}"
os.makedirs(plot_dir, exist_ok=True)

top = feat_imp.head(TOP_N).iloc[::-1]  # Reverse for horizontal bar
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top["Feature"], top["Importance"], color="steelblue", edgecolor="black")
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title(f"{TASK_NAME}: Top {TOP_N} Magpie Feature Importances (RF, Fold {FOLD_IDX})",
             fontsize=13)
plt.tight_layout()

plot_path = f"{plot_dir}/feature_importance_fold{FOLD_IDX}.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"  Saved: {plot_path}")

# Save full feature importance table
csv_path = f"{plot_dir}/feature_importance_fold{FOLD_IDX}.csv"
feat_imp.to_csv(csv_path, index=False)
print(f"  Full table saved: {csv_path}")

print("\nDone!")
```

## Script 4: PyTorch MLP on Magpie Features

Bridge from sklearn to neural approaches: a 3-layer MLP trained on Magpie features with GPU support.

```python
#!/opt/conda/envs/matbench/bin/python
"""
PyTorch MLP on Magpie Features: Composition-based neural baseline for MatBench.
3-layer MLP with BatchNorm, Dropout, GPU auto-detection.
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.makedirs("/workspace/group/matbench/data", exist_ok=True)
os.makedirs("/workspace/group/matbench/results", exist_ok=True)
os.makedirs("/workspace/group/matbench/models", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"  # Composition-input task
IS_CLASSIFICATION = False
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20  # Early stopping patience
# ======================

# GPU auto-detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"=" * 70)
print(f"PyTorch MLP + Magpie | {TASK_NAME}")
print(f"=" * 70)

# Featurizer
featurizer = ElementProperty.from_preset("magpie")

def featurize_compositions(compositions):
    comp_objects = [Composition(c) if isinstance(c, str) else c for c in compositions]
    df_comp = pd.DataFrame({"composition": comp_objects})
    print(f"    Featurizing {len(comp_objects)} compositions...")
    df_feat = featurizer.featurize_dataframe(df_comp, "composition", ignore_errors=True)
    df_feat = df_feat.drop(columns=["composition"])
    df_feat = df_feat.fillna(0)
    return df_feat


class MagpieMLP(nn.Module):
    """3-layer MLP with BatchNorm and Dropout for Magpie features."""

    def __init__(self, input_dim, hidden_dims, dropout, is_classification):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        # Output layer
        if is_classification:
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, input_dim):
    """Train MLP with early stopping."""
    # Split train into train/val (90/10) for early stopping
    n_val = max(1, int(0.1 * len(X_train)))
    indices = np.random.RandomState(42).permutation(len(X_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr = torch.FloatTensor(X_train[train_idx]).to(device)
    y_tr = torch.FloatTensor(y_train[train_idx]).to(device)
    X_va = torch.FloatTensor(X_train[val_idx]).to(device)
    y_va = torch.FloatTensor(y_train[val_idx]).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MagpieMLP(input_dim, HIDDEN_DIMS, DROPOUT, IS_CLASSIFICATION).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if IS_CLASSIFICATION:
        criterion = nn.BCELoss()
    else:
        criterion = nn.L1Loss()  # MAE

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = criterion(val_pred, y_va).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# Load task
print("\nLoading task...")
mb = MatbenchBenchmark(autoload=False)
task = None
for t in mb.tasks:
    if t.dataset_name == TASK_NAME:
        task = t
        break
task.load()
print(f"  Loaded: {task.dataset_name} ({len(task.df)} samples)")

# 5-fold cross-validation
fold_scores = []

for fold_idx in range(5):
    print(f"\n--- Fold {fold_idx} ---")

    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    print("  Featurizing train set...")
    X_train_df = featurize_compositions(train_inputs)
    print("  Featurizing test set...")
    X_test_df = featurize_compositions(test_inputs)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)
    y_train = train_outputs.values.astype(np.float32)

    input_dim = X_train_scaled.shape[1]

    # Train
    print(f"  Training MLP (dims={HIDDEN_DIMS}, lr={LEARNING_RATE}, epochs={EPOCHS})...")
    model = train_mlp(X_train_scaled, y_train, input_dim)

    # Predict
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    # Record
    task.record(fold_idx, predictions)

    # Save model checkpoint
    ckpt_path = f"/workspace/group/matbench/models/{TASK_NAME}_MLP_fold{fold_idx}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Score
    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        score = roc_auc_score(true_values, predictions)
        metric = "ROC-AUC"
    else:
        score = mean_absolute_error(true_values, predictions)
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: MLP + Magpie on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results_path = f"/workspace/group/matbench/results/{TASK_NAME}_MLP_Magpie.json"
results = {
    "task": TASK_NAME,
    "model": "MLP_Magpie",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "hidden_dims": HIDDEN_DIMS,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "device": str(device),
}
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
print("Done!")
```

## Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `n_estimators` | Number of trees (RF/GBR) | 500 | More trees = better but slower |
| `max_depth` | Max tree depth (GBR) | 6 | Higher = more complex, risk overfitting |
| `learning_rate` | Step size (GBR / MLP) | 0.05 / 1e-3 | Lower = slower convergence, often better |
| `Magpie preset` | Featurizer preset | `"magpie"` | 132 elemental property features |
| `n_jobs` | Parallel jobs (sklearn) | -1 | -1 = use all cores |
| `nan_strategy` | How to handle NaN features | `fillna(0)` | Can also use median imputation |
| `HIDDEN_DIMS` | MLP hidden layer sizes | [256, 128, 64] | Adjust based on dataset size |
| `DROPOUT` | MLP dropout rate | 0.2 | Increase for small datasets |
| `PATIENCE` | Early stopping patience | 20 | Epochs without improvement |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| NaN in Magpie features | Some compositions have elements not in Magpie | Use `ignore_errors=True` in featurizer + `fillna(0)` |
| `ConvergenceWarning` (GBR) | Too few estimators or learning rate too high | Increase `n_estimators`, decrease `learning_rate` |
| Poor MAE on small data (steels) | Only 312 samples, models overfit easily | Use RF (not GBR), reduce `max_depth`, increase regularization |
| Featurization slow on large tasks | ElementProperty computes many statistics | Cache featurized DataFrames to `/workspace/group/matbench/data/` |
| Classification predictions as float | MatBench expects probabilities for classification | Use `predict_proba()[:, 1]`, not `predict()` |
| `KeyError: composition` | Input column name mismatch | MatBench returns a Series; wrap in DataFrame with column name `"composition"` |
| GPU out of memory (MLP) | Batch size too large | Reduce `BATCH_SIZE` to 32 or 16 |
| MLP not converging | Learning rate too high or features not scaled | Always use `StandardScaler`; try lower learning rate |
