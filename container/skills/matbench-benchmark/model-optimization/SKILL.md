# Deep Learning Optimization for MatBench SOTA

## When to Use This Skill

This skill should be used when:
- You have a working deep learning baseline and want to push toward SOTA performance
- You want to ensemble multiple deep learning models for improved accuracy
- You need transfer learning from materials science foundation models
- You want systematic neural architecture search and hyperparameter optimization
- You need advanced PyTorch training tricks (SWA, EMA, mixup, cosine annealing)
- You want a complete phased roadmap for achieving top leaderboard positions

## Method Selection

```
What optimization strategy?

Quick boost (any DL model)?
  -> Script 1: Multi-seed deep ensemble (average 3-5 models with different seeds)

Combine diverse architectures?
  -> Script 2: Cross-architecture deep ensemble (CGCNN + Roost + MACE-MLP)

Small dataset or leverage pretrained weights?
  -> Script 3: Transfer learning from foundation models (MACE-MP-0, CHGNet, M3GNet)

Systematic architecture/hyperparam search?
  -> Script 4: Neural architecture search with Optuna (GPU-accelerated)

Squeeze last few % from a strong model?
  -> Script 5: Advanced training tricks (SWA, EMA, mixup, cosine annealing)

Full SOTA pursuit?
  -> Section 6: Complete SOTA strategy roadmap (phased plan)
```

## Prerequisites

- MatBench conda environment with torch, torch-geometric, matbench, pymatgen
- A working deep learning baseline (see structure-gnn/ or composition-models/ skills)
- GPU recommended (NVIDIA A100-SXM4-80GB available)
- Foundation model weights for transfer learning (MACE-MP-0, CHGNet, M3GNet)

## Script 1: Multi-Seed Deep Ensemble

Train the same deep learning architecture N times with different random seeds and average predictions. This is the simplest and most reliable way to improve results, typically yielding 5-15% MAE improvement through variance reduction.

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `N_SEEDS` | 5 | More seeds = better but slower; 3-5 is typical |
| `EPOCHS` | 300 | Per-seed training epochs |
| `LR` | 1e-3 | Learning rate for each member |
| `BATCH_SIZE` | 64 | Adjust for GPU memory |

```python
#!/opt/conda/envs/matbench/bin/python
"""
Multi-seed deep ensemble: train N identical DL models with different seeds,
average predictions for variance reduction.
Typical improvement: 5-15% MAE reduction over single model.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
TASK_NAME = "matbench_mp_e_form"  # Change as needed
N_SEEDS = 5                       # Number of ensemble members
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64

CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS

# Output directory
timestamp = datetime.now().strftime("%Y-%m-%d")
out_dir = f"/workspace/group/matbench/experiments/{timestamp}_multiseed_ensemble_{TASK_NAME}"
os.makedirs(out_dir, exist_ok=True)

mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

seed_metrics = {seed: [] for seed in range(N_SEEDS)}
ensemble_metrics = []

for task in mb.tasks:
    task.load()
    print(f"\nTask: {task.dataset_name}")
    print(f"Ensemble size: {N_SEEDS} seeds")

    for fold_idx in task.folds:
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx}")
        print(f"{'='*50}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

        # === USER: Replace this section with your actual model + featurizer ===
        # This is a placeholder showing the ensemble pattern.
        # Swap in your GNN DataLoader, model class, training loop, etc.

        all_predictions = []

        for seed in range(N_SEEDS):
            print(f"\n  --- Seed {seed} ---")

            # Set all random seeds deterministically
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            # --- Build your DL model here ---
            # model = YourGNNModel(...).to(device)
            # optimizer = optim.Adam(model.parameters(), lr=LR)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, ...)
            # ... training loop over EPOCHS ...
            # preds = model.predict(test_loader)

            # Placeholder: replace with actual model predictions
            # all_predictions.append(preds)

            print(f"    Seed {seed} training complete")

        # --- Ensemble averaging ---
        # pred_array = np.stack(all_predictions, axis=0)  # (N_SEEDS, n_test)
        # if is_classification:
        #     ensemble_preds = (pred_array.mean(axis=0) > 0.5).astype(bool)
        # else:
        #     ensemble_preds = pred_array.mean(axis=0)

        # Per-seed std shows prediction uncertainty
        # pred_std = pred_array.std(axis=0)
        # print(f"  Mean prediction std: {pred_std.mean():.4f}")

        # task.record(fold_idx, ensemble_preds)
        print(f"  Fold {fold_idx} ensemble complete")

    # Save results
    # results = mb.get_results()
    # with open(os.path.join(out_dir, "results.json"), "w") as f:
    #     json.dump(results, f, indent=2)

print(f"\nResults saved to {out_dir}")
```

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM with many seeds | Reduce `BATCH_SIZE` or train seeds sequentially, deleting model between seeds |
| Diminishing returns past 5 seeds | Normal -- variance reduction scales as 1/sqrt(N) |
| Seeds produce very similar predictions | Increase `LR` spread or use dropout for diversity |
| GPU underutilized | Use `torch.compile(model)` on PyTorch 2.x for speedup |

---

## Script 2: Cross-Architecture Deep Ensemble

Combine predictions from multiple distinct deep learning architectures (e.g., CGCNN, Roost, MACE-MLP). Architecture diversity provides complementary error patterns that simple seed ensembles cannot capture.

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ARCHITECTURES` | `["cgcnn", "roost", "mace_mlp"]` | List of DL model types |
| `WEIGHTING` | `"learned"` | `"equal"`, `"learned"`, or `"val_performance"` |
| `META_EPOCHS` | 50 | Epochs for training the learned weighting MLP |

```python
#!/opt/conda/envs/matbench/bin/python
"""
Cross-architecture deep ensemble: combine CGCNN, Roost, MACE-MLP (or others).
Uses learned weighting or simple averaging of predictions.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
TASK_NAME = "matbench_mp_e_form"
ARCHITECTURES = ["cgcnn", "roost", "mace_mlp"]
WEIGHTING = "learned"  # "equal", "learned", or "val_performance"

timestamp = datetime.now().strftime("%Y-%m-%d")
out_dir = f"/workspace/group/matbench/experiments/{timestamp}_cross_arch_ensemble_{TASK_NAME}"
os.makedirs(out_dir, exist_ok=True)

CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS

mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

for task in mb.tasks:
    task.load()

    for fold_idx in task.folds:
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx}")
        print(f"{'='*50}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        arch_preds = {}
        arch_val_scores = {}

        for arch_name in ARCHITECTURES:
            print(f"\n  Training {arch_name}...")

            # === USER: Load/train each architecture ===
            # if arch_name == "cgcnn":
            #     model = CGCNN(...).to(device)
            # elif arch_name == "roost":
            #     model = Roost(...).to(device)
            # elif arch_name == "mace_mlp":
            #     model = MACE_MLP(...).to(device)
            #
            # ... train model ...
            # val_preds = model.predict(val_loader)
            # val_score = compute_metric(val_targets, val_preds)
            # arch_val_scores[arch_name] = val_score
            #
            # test_preds = model.predict(test_loader)
            # arch_preds[arch_name] = test_preds

            print(f"    {arch_name} training complete")

        # --- Combine predictions ---
        # pred_matrix = np.stack([arch_preds[a] for a in ARCHITECTURES], axis=0)
        #
        # if WEIGHTING == "equal":
        #     weights = np.ones(len(ARCHITECTURES)) / len(ARCHITECTURES)
        #     ensemble_preds = (pred_matrix * weights[:, None]).sum(axis=0)
        #
        # elif WEIGHTING == "val_performance":
        #     # Weight inversely proportional to validation error
        #     errors = np.array([arch_val_scores[a] for a in ARCHITECTURES])
        #     weights = (1.0 / errors) / (1.0 / errors).sum()
        #     ensemble_preds = (pred_matrix * weights[:, None]).sum(axis=0)
        #
        # elif WEIGHTING == "learned":
        #     # Train a small MLP to learn optimal combination
        #     # Use validation set predictions as features, val targets as labels
        #     class WeightNet(nn.Module):
        #         def __init__(self, n_models):
        #             super().__init__()
        #             self.net = nn.Sequential(
        #                 nn.Linear(n_models, 16),
        #                 nn.ReLU(),
        #                 nn.Linear(16, 1)
        #             )
        #         def forward(self, x):
        #             return self.net(x).squeeze(-1)
        #
        #     weight_net = WeightNet(len(ARCHITECTURES)).to(device)
        #     opt = optim.Adam(weight_net.parameters(), lr=1e-3)
        #     # ... train weight_net on val predictions -> val targets ...
        #     # ensemble_preds = weight_net(test_pred_tensor).detach().cpu().numpy()
        #
        # if is_classification:
        #     ensemble_preds = ensemble_preds > 0.5
        #
        # task.record(fold_idx, ensemble_preds)

    # Save
    # results = mb.get_results()
    # with open(os.path.join(out_dir, "results.json"), "w") as f:
    #     json.dump(results, f, indent=2)

print(f"\nResults saved to {out_dir}")
```

### Per-Task Recommended Architectures

| Task Category | Recommended Combination |
|---------------|------------------------|
| Structure-based regression | CGCNN + MEGNet + MACE-MLP |
| Structure-based classification | CGCNN + SchNet + DimeNet++ |
| Composition-only regression | Roost + CrabNet + ElemNet |
| Composition-only classification | Roost + CrabNet |

### Common Issues

| Issue | Solution |
|-------|----------|
| One architecture dominates | Use learned weighting; it will down-weight weak models |
| Architectures produce correlated errors | Add architecturally diverse models (e.g., message-passing + attention) |
| Learned weighting overfits | Use val fold; reduce META_EPOCHS; add weight decay |
| Very different prediction scales | Normalize predictions to zero mean / unit variance before combining |

---

## Script 3: Transfer Learning from Foundation Models

Use pretrained materials science foundation models (MACE-MP-0, CHGNet, M3GNet) as feature extractors or for fine-tuning. Especially effective on small-data matbench tasks.

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `FOUNDATION_MODEL` | `"mace-mp-0"` | `"mace-mp-0"`, `"chgnet"`, or `"m3gnet"` |
| `STRATEGY` | `"finetune_last_n"` | `"freeze_extract"`, `"finetune_last_n"`, `"full_finetune"` |
| `N_UNFREEZE_LAYERS` | 3 | Layers to unfreeze from the end |
| `HEAD_LR` | 1e-3 | Learning rate for new MLP head |
| `BACKBONE_LR` | 1e-5 | Learning rate for unfrozen backbone layers |

```python
#!/opt/conda/envs/matbench/bin/python
"""
Transfer learning from materials science foundation models.
Strategies:
  1. freeze_extract: Freeze backbone, train MLP head only
  2. finetune_last_n: Unfreeze last N layers with discriminative LR
  3. full_finetune: Unfreeze all with very low backbone LR
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
TASK_NAME = "matbench_jdft2d"  # Small dataset benefits most from transfer learning
FOUNDATION_MODEL = "mace-mp-0"  # "mace-mp-0", "chgnet", "m3gnet"
STRATEGY = "finetune_last_n"    # "freeze_extract", "finetune_last_n", "full_finetune"
N_UNFREEZE_LAYERS = 3
HEAD_LR = 1e-3
BACKBONE_LR = 1e-5
EPOCHS = 200
BATCH_SIZE = 32

timestamp = datetime.now().strftime("%Y-%m-%d")
out_dir = f"/workspace/group/matbench/experiments/{timestamp}_transfer_{FOUNDATION_MODEL}_{TASK_NAME}"
os.makedirs(out_dir, exist_ok=True)

CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS


def load_foundation_model(model_name):
    """Load a pretrained foundation model."""
    if model_name == "mace-mp-0":
        # from mace.calculators import mace_mp
        # model = mace_mp(model="medium", device=str(device))
        # backbone = model.model  # extract the torch module
        pass
    elif model_name == "chgnet":
        # from chgnet.model import CHGNet
        # backbone = CHGNet.load().model
        pass
    elif model_name == "m3gnet":
        # from matgl.ext.ase import M3GNetCalculator
        # import matgl
        # pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        # backbone = pot.model
        pass
    else:
        raise ValueError(f"Unknown foundation model: {model_name}")
    # return backbone
    return None


class TransferModel(nn.Module):
    """Wraps a foundation model backbone with a task-specific MLP head."""

    def __init__(self, backbone, feature_dim, output_dim=1, is_clf=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )
        self.is_clf = is_clf

    def forward(self, batch):
        # Extract features from backbone (model-specific)
        features = self.backbone.extract_features(batch)  # shape: (batch, feature_dim)
        out = self.head(features)
        if self.is_clf:
            out = torch.sigmoid(out)
        return out.squeeze(-1)


def setup_discriminative_lr(model, strategy, head_lr, backbone_lr, n_unfreeze):
    """Configure parameter groups with discriminative learning rates."""
    if strategy == "freeze_extract":
        # Freeze entire backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        param_groups = [{"params": model.head.parameters(), "lr": head_lr}]

    elif strategy == "finetune_last_n":
        # Freeze all backbone layers, then unfreeze last N
        for param in model.backbone.parameters():
            param.requires_grad = False

        backbone_layers = list(model.backbone.children())
        layers_to_unfreeze = backbone_layers[-n_unfreeze:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Discriminative LR: backbone gets lower LR, head gets higher LR
        unfrozen_backbone_params = [
            p for layer in layers_to_unfreeze for p in layer.parameters() if p.requires_grad
        ]
        param_groups = [
            {"params": unfrozen_backbone_params, "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]

    elif strategy == "full_finetune":
        # All parameters trainable, backbone at lower LR
        param_groups = [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]

    return param_groups


# --- Main training loop ---
mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

for task in mb.tasks:
    task.load()
    print(f"\nTask: {task.dataset_name}")
    print(f"Foundation model: {FOUNDATION_MODEL}")
    print(f"Strategy: {STRATEGY}")

    for fold_idx in task.folds:
        print(f"\nFold {fold_idx}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        # === USER: Implement actual model loading and training ===
        # backbone = load_foundation_model(FOUNDATION_MODEL)
        # model = TransferModel(backbone, feature_dim=128, is_clf=is_classification).to(device)
        #
        # param_groups = setup_discriminative_lr(
        #     model, STRATEGY, HEAD_LR, BACKBONE_LR, N_UNFREEZE_LAYERS
        # )
        # optimizer = optim.AdamW(param_groups, weight_decay=1e-5)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        #
        # criterion = nn.BCELoss() if is_classification else nn.L1Loss()
        #
        # for epoch in range(EPOCHS):
        #     model.train()
        #     for batch in train_loader:
        #         batch = batch.to(device)
        #         optimizer.zero_grad()
        #         pred = model(batch)
        #         loss = criterion(pred, batch.y)
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        #         optimizer.step()
        #         scheduler.step(epoch + batch_idx / len(train_loader))
        #
        # model.eval()
        # with torch.no_grad():
        #     preds = model(test_data)
        # task.record(fold_idx, preds.cpu().numpy())

        print(f"  Fold {fold_idx} complete")

print(f"\nResults saved to {out_dir}")
```

### Strategy Selection Guide

| Dataset Size | Recommended Strategy | Notes |
|-------------|---------------------|-------|
| < 1000 samples | `freeze_extract` | Avoid overfitting the backbone |
| 1000-10000 | `finetune_last_n` (N=2-4) | Best balance of adaptation and stability |
| > 10000 | `full_finetune` | Enough data to adapt the full model |

### Common Issues

| Issue | Solution |
|-------|----------|
| Backbone features are poor | Try a different foundation model or increase `N_UNFREEZE_LAYERS` |
| Fine-tuning destroys pretrained knowledge | Lower `BACKBONE_LR`; use warmup; freeze more layers |
| Head overfits on small data | Add dropout; reduce head width; use weight decay |
| Foundation model not installed | `pip install mace-torch` / `pip install chgnet` / `pip install matgl` |

---

## Script 4: Neural Architecture Search (Lightweight)

GPU-accelerated random + Optuna-based hyperparameter optimization for deep learning models. Searches over hidden dimensions, number of layers, attention heads, dropout, activation functions, and learning rate schedules.

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `N_TRIALS` | 50 | Total Optuna trials |
| `EPOCHS_PER_TRIAL` | 100 | Reduced epochs for faster search |
| `PRUNING` | `True` | Early stop bad trials via MedianPruner |
| `SEARCH_SPACE` | see below | Defines the hyperparameter ranges |

```python
#!/opt/conda/envs/matbench/bin/python
"""
Lightweight neural architecture search with Optuna.
Searches over DL hyperparameters: hidden_dim, n_layers, n_heads, dropout,
activation, learning rate, weight decay, scheduler.
Uses GPU-accelerated trials with optional pruning.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.pruners import MedianPruner
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
TASK_NAME = "matbench_mp_e_form"
N_TRIALS = 50
EPOCHS_PER_TRIAL = 100
PRUNING = True

timestamp = datetime.now().strftime("%Y-%m-%d")
out_dir = f"/workspace/group/matbench/experiments/{timestamp}_nas_{TASK_NAME}"
os.makedirs(out_dir, exist_ok=True)

CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS


def build_model(trial):
    """Build a model from Optuna-suggested hyperparameters."""
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    activation = trial.suggest_categorical("activation", ["ReLU", "SiLU", "GELU", "Mish"])

    act_fn = getattr(nn, activation)()

    # === USER: Replace with your actual GNN/transformer architecture ===
    # Example: configurable MLP (replace with GNN for structure tasks)
    layers = []
    in_dim = 128  # feature dimension from your featurizer
    for i in range(n_layers):
        out_dim = hidden_dim if i < n_layers - 1 else 1
        layers.append(nn.Linear(in_dim, out_dim))
        if i < n_layers - 1:
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    model = nn.Sequential(*layers).to(device)
    return model


def objective(trial):
    """Single Optuna trial: build model, train, return validation metric."""
    model = build_model(trial)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "plateau", "step"])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    elif scheduler_type == "step":
        step_size = trial.suggest_int("step_size", 20, 80, step=10)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    criterion = nn.BCELoss() if is_classification else nn.L1Loss()

    # === USER: Replace with actual training loop ===
    # for epoch in range(EPOCHS_PER_TRIAL):
    #     model.train()
    #     train_loss = 0
    #     for batch in train_loader:
    #         batch = batch.to(device)
    #         optimizer.zero_grad()
    #         pred = model(batch)
    #         loss = criterion(pred, batch.y)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #
    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         val_metric = evaluate(model, val_loader)
    #
    #     if scheduler_type == "plateau":
    #         scheduler.step(val_metric)
    #     else:
    #         scheduler.step()
    #
    #     # Report to Optuna for pruning
    #     trial.report(val_metric, epoch)
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()
    #
    # return val_metric  # minimize MAE or maximize AUC

    return 0.0  # placeholder


# Run the study
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20) if PRUNING else None
study = optuna.create_study(
    direction="minimize",  # "maximize" for classification AUC
    pruner=pruner,
    study_name=f"nas_{TASK_NAME}",
)
# study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# Report best trial
# print(f"\nBest trial:")
# print(f"  Value: {study.best_trial.value:.6f}")
# print(f"  Params: {study.best_trial.params}")
#
# # Save study results
# results = {
#     "best_value": study.best_trial.value,
#     "best_params": study.best_trial.params,
#     "all_trials": [
#         {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
#         for t in study.trials
#     ],
# }
# with open(os.path.join(out_dir, "nas_results.json"), "w") as f:
#     json.dump(results, f, indent=2)
#
# # Plot optimization history
# fig = optuna.visualization.matplotlib.plot_optimization_history(study)
# plt.savefig(os.path.join(out_dir, "optimization_history.png"), dpi=150, bbox_inches="tight")
# plt.close()
#
# fig = optuna.visualization.matplotlib.plot_param_importances(study)
# plt.savefig(os.path.join(out_dir, "param_importances.png"), dpi=150, bbox_inches="tight")
# plt.close()

print(f"\nNAS results saved to {out_dir}")
```

### Search Space Summary

| Hyperparameter | Range | Scale |
|---------------|-------|-------|
| `hidden_dim` | {64, 128, 256, 512} | Categorical |
| `n_layers` | 2 - 6 | Integer |
| `n_heads` | {1, 2, 4, 8} | Categorical |
| `dropout` | 0.0 - 0.5 | Step 0.05 |
| `activation` | {ReLU, SiLU, GELU, Mish} | Categorical |
| `lr` | 1e-5 - 1e-2 | Log-uniform |
| `weight_decay` | 1e-6 - 1e-2 | Log-uniform |
| `scheduler` | {cosine, plateau, step} | Categorical |

### Common Issues

| Issue | Solution |
|-------|----------|
| Trials OOM on GPU | Reduce max `hidden_dim` or `n_layers` in search space |
| Too many pruned trials | Increase `n_warmup_steps` in MedianPruner |
| Best params overfit | Validate best params on held-out fold; reduce `EPOCHS_PER_TRIAL` |
| Optuna not installed | `pip install optuna optuna-dashboard` |

---

## Script 5: Advanced Training Tricks for SOTA

Collection of advanced PyTorch training techniques that individually provide 1-5% improvement, and stack for significant cumulative gains.

### Techniques Covered

| Technique | Typical Gain | Best For |
|-----------|-------------|----------|
| Cosine Annealing + Warm Restarts | 2-5% | All tasks |
| Stochastic Weight Averaging (SWA) | 2-4% | Regression tasks |
| Exponential Moving Average (EMA) | 1-3% | Noisy training |
| Label Smoothing | 1-3% | Classification tasks |
| Mixup Augmentation | 2-5% | Small datasets |
| Gradient Accumulation | Enables large batch | GPU memory limited |

```python
#!/opt/conda/envs/matbench/bin/python
"""
Advanced PyTorch training tricks for SOTA performance.
Implements: cosine annealing, SWA, EMA, label smoothing, mixup,
gradient accumulation -- all in one configurable training loop.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import copy
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
TASK_NAME = "matbench_mp_e_form"
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64

# Toggle techniques
USE_COSINE_ANNEALING = True
USE_SWA = True
SWA_START_EPOCH = 200       # Start SWA after this epoch
SWA_LR = 5e-4               # SWA learning rate

USE_EMA = True
EMA_DECAY = 0.999            # EMA decay factor

USE_LABEL_SMOOTHING = True   # For classification tasks only
LABEL_SMOOTHING = 0.1

USE_MIXUP = True
MIXUP_ALPHA = 0.2            # Beta distribution parameter

USE_GRAD_ACCUMULATION = True
ACCUMULATION_STEPS = 4       # Effective batch = BATCH_SIZE * ACCUMULATION_STEPS

timestamp = datetime.now().strftime("%Y-%m-%d")
out_dir = f"/workspace/group/matbench/experiments/{timestamp}_advanced_tricks_{TASK_NAME}"
os.makedirs(out_dir, exist_ok=True)

CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS


# --- Exponential Moving Average ---
class EMAModel:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Swap model params with EMA params for inference."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params after inference."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


# --- Mixup ---
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- Main training loop with all tricks ---
mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

for task in mb.tasks:
    task.load()
    print(f"\nTask: {task.dataset_name}")
    print(f"Techniques: cosine={USE_COSINE_ANNEALING}, SWA={USE_SWA}, "
          f"EMA={USE_EMA}, mixup={USE_MIXUP}, grad_accum={USE_GRAD_ACCUMULATION}")

    for fold_idx in task.folds:
        print(f"\nFold {fold_idx}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        # === USER: Build your model and data loaders ===
        # model = YourModel(...).to(device)
        # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Optimizer
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

        # Cosine annealing with warm restarts
        # if USE_COSINE_ANNEALING:
        #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer, T_0=50, T_mult=2, eta_min=1e-6
        #     )

        # SWA model
        # if USE_SWA:
        #     swa_model = AveragedModel(model).to(device)
        #     swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

        # EMA
        # if USE_EMA:
        #     ema = EMAModel(model, decay=EMA_DECAY)

        # Loss function
        # if is_classification and USE_LABEL_SMOOTHING:
        #     criterion = nn.BCEWithLogitsLoss(
        #         # Label smoothing via soft targets
        #     )
        # elif is_classification:
        #     criterion = nn.BCEWithLogitsLoss()
        # else:
        #     criterion = nn.L1Loss()  # MAE for regression

        # === Training loop ===
        # for epoch in range(EPOCHS):
        #     model.train()
        #     optimizer.zero_grad()
        #
        #     for batch_idx, batch in enumerate(train_loader):
        #         batch = batch.to(device)
        #
        #         # Mixup augmentation
        #         if USE_MIXUP and not is_classification:
        #             mixed_x, y_a, y_b, lam = mixup_data(batch.x, batch.y, MIXUP_ALPHA)
        #             pred = model(mixed_x)
        #             loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
        #         else:
        #             pred = model(batch)
        #             # Label smoothing for classification
        #             if is_classification and USE_LABEL_SMOOTHING:
        #                 smooth_y = batch.y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        #                 loss = criterion(pred, smooth_y)
        #             else:
        #                 loss = criterion(pred, batch.y)
        #
        #         # Gradient accumulation
        #         if USE_GRAD_ACCUMULATION:
        #             loss = loss / ACCUMULATION_STEPS
        #
        #         loss.backward()
        #
        #         if USE_GRAD_ACCUMULATION:
        #             if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
        #                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        #                 optimizer.step()
        #                 optimizer.zero_grad()
        #         else:
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        #             optimizer.step()
        #             optimizer.zero_grad()
        #
        #     # Scheduler step
        #     if USE_SWA and epoch >= SWA_START_EPOCH:
        #         swa_model.update_parameters(model)
        #         swa_scheduler.step()
        #     elif USE_COSINE_ANNEALING:
        #         scheduler.step()
        #
        #     # EMA update
        #     if USE_EMA:
        #         ema.update(model)
        #
        #     if epoch % 50 == 0:
        #         print(f"    Epoch {epoch}, loss: {loss.item():.4f}")

        # === Prediction: choose best model variant ===
        # model.eval()
        #
        # # SWA: update batch norm statistics
        # if USE_SWA:
        #     torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        #     swa_model.eval()
        #     with torch.no_grad():
        #         swa_preds = swa_model(test_data)
        #
        # # EMA prediction
        # if USE_EMA:
        #     ema.apply_shadow(model)
        #     with torch.no_grad():
        #         ema_preds = model(test_data)
        #     ema.restore(model)
        #
        # # Regular model prediction
        # with torch.no_grad():
        #     base_preds = model(test_data)
        #
        # # Average SWA, EMA, and base predictions for best result
        # final_preds = (swa_preds + ema_preds + base_preds) / 3
        # task.record(fold_idx, final_preds.cpu().numpy())

        print(f"  Fold {fold_idx} complete")

print(f"\nResults saved to {out_dir}")
```

### Common Issues

| Issue | Solution |
|-------|----------|
| SWA degrades performance | Start SWA later (after LR has decayed); tune `SWA_LR` |
| Mixup hurts on very small datasets | Reduce `MIXUP_ALPHA` to 0.1 or disable |
| EMA too aggressive | Increase `EMA_DECAY` closer to 1.0 (e.g., 0.9999) |
| Gradient accumulation wrong loss scale | Ensure `loss / ACCUMULATION_STEPS` before `.backward()` |
| Cosine schedule restarts too frequent | Increase `T_0` or `T_mult` |
| Label smoothing on regression | Not applicable; only use for classification tasks |

---

## Section 6: Complete SOTA Strategy Roadmap

A phased plan for systematically achieving top matbench leaderboard positions using deep learning.

### Phase 1: Study Reference Repos (1-2 days)

1. **Review the reference library** (`reference-library/SKILL.md`) for SOTA model repos
2. Identify the top 3 architectures for your target task category:
   - Structure-based: MACE, DimeNet++, CGCNN, MEGNet, SchNet
   - Composition-based: Roost, CrabNet, ElemNet
3. Read their papers, note training details: LR, scheduler, epochs, data splits
4. Check if pretrained weights are available

### Phase 2: Baseline with Best Known Architecture (2-3 days)

1. Implement the highest-performing known architecture for your task
2. Use the **exact** hyperparameters from the original paper
3. Run full 5-fold matbench evaluation
4. Record baseline MAE/AUC -- this is your reference point
5. Verify reproducibility: run twice, check consistency

### Phase 3: Hyperparameter Optimization (2-3 days)

1. Run **Script 4** (Optuna NAS) to search over:
   - Model architecture params (hidden_dim, n_layers, n_heads, dropout)
   - Training params (LR, weight_decay, scheduler type)
2. Use fold 0 for initial search (fast iteration)
3. Validate top-3 configs across all 5 folds
4. Expected improvement: 5-15% over paper defaults

### Phase 4: Ensemble + Advanced Tricks (2-3 days)

1. Apply **Script 5** (advanced tricks) to the optimized architecture:
   - SWA + EMA typically give 2-5% improvement
   - Cosine annealing with warm restarts for stable convergence
   - Mixup for small-data tasks
2. Run **Script 1** (multi-seed ensemble) with 5 seeds: ~5-10% improvement
3. Run **Script 2** (cross-architecture ensemble) combining top 2-3 architectures
4. If structure data available, apply **Script 3** (transfer learning) with MACE-MP-0

### Phase 5: Submit and Compare (1 day)

1. Generate final `results.json` from matbench benchmark
2. Compare against leaderboard at [matbench leaderboard](https://matbench.materialsproject.org/)
3. Record all hyperparameters, ensemble configs, and training details
4. Save the full experiment log:

```
/workspace/group/matbench/experiments/
  YYYY-MM-DD_sota_attempt_taskname/
    results.json            # matbench official results
    config.yaml             # all hyperparameters
    training_log.csv        # epoch-by-epoch metrics
    model_checkpoints/      # saved .pt files
    plots/                  # learning curves, parity plots
    README.md               # human-readable experiment summary
```

### Expected Cumulative Improvement

| Stage | Typical Improvement | Running Total |
|-------|-------------------|---------------|
| Paper baseline | -- | 100% (reference) |
| Hyperparameter opt | 5-15% | 85-95% of baseline error |
| SWA + EMA | 2-5% | 80-93% |
| Multi-seed ensemble (5x) | 5-10% | 72-88% |
| Cross-arch ensemble | 3-8% | 66-85% |
| Transfer learning | 2-10% (task dependent) | 60-83% |

Note: percentages are multiplicative error reduction; actual gains vary by task and how close the baseline is to the theoretical floor.
