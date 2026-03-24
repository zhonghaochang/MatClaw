# Optimize MatBench Model Performance for SOTA

## When to Use This Skill

This skill should be used when:
- You have a working baseline model and want to push toward SOTA performance
- You want to ensemble multiple models for improved accuracy
- You need transfer learning strategies for small dataset tasks
- You want systematic hyperparameter optimization
- You need data augmentation or feature engineering strategies
- You want a complete roadmap for achieving top leaderboard positions

## Method Selection

```
What optimization strategy?

Quick boost (any model)?
  → Script 1: Multi-seed ensemble (average 3-5 models with different seeds)

Small dataset (steels, jdft2d, phonons)?
  → Script 3: Transfer learning from large-task pretrained models
  → Script 5: Feature engineering

Large dataset, want best accuracy?
  → Script 4: Architecture search + hyperparameter sweep
  → Script 2: Cross-architecture ensemble

Classification optimization?
  → Script 2: Ensemble with probability averaging + threshold tuning

Systematic SOTA pursuit?
  → Section 6: Complete SOTA strategy guide (phased roadmap)
```

## Prerequisites

- MatBench conda environment with torch, torch-geometric, matbench, matminer, scikit-learn
- A working baseline model (see structure-gnn/ or composition-models/ skills)
- GPU recommended (NVIDIA A100-SXM4-80GB available)

## Script 1: Multi-Seed Ensemble

Train the same model N times with different random seeds and average predictions. This is the simplest and most reliable way to improve results, typically yielding 5-15% MAE improvement.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Multi-seed ensemble: train N identical models with different seeds,
average predictions for improved robustness.
Typical improvement: 5-15% MAE reduction over single model.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

# Classification tasks use probability averaging
CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS

mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

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

        all_predictions = []

        for seed in range(N_SEEDS):
            print(f"\n  --- Seed {seed} ---")

            # Set all random seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # ========================================
            # YOUR MODEL INITIALIZATION HERE
            # Example with a generic model:
            # model = CGCNN(config).to(device)
            # ========================================

            # Placeholder: replace with your actual model training
            # For demonstration, we show the ensemble logic:
            #
            # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            #
            # for epoch in range(EPOCHS):
            #     model.train()
            #     for batch in train_loader:
            #         optimizer.zero_grad()
            #         pred = model(batch)
            #         loss = criterion(pred, batch.y)
            #         loss.backward()
            #         optimizer.step()
            #     scheduler.step()
            #     if (epoch + 1) % 50 == 0:
            #         print(f"    Epoch {epoch+1}/{EPOCHS}")
            #
            # model.eval()
            # with torch.no_grad():
            #     seed_preds = model.predict(test_data)

            # Placeholder predictions (replace with actual)
            seed_preds = np.zeros(len(test_inputs))

            all_predictions.append(seed_preds)
            print(f"    Seed {seed} predictions: mean={seed_preds.mean():.4f}, std={seed_preds.std():.4f}")

            # Save individual model
            ckpt_dir = f"/workspace/group/matbench/models/ensemble/{TASK_NAME}/fold_{fold_idx}"
            os.makedirs(ckpt_dir, exist_ok=True)
            # torch.save(model.state_dict(), f"{ckpt_dir}/seed_{seed}.pt")

        # Ensemble: average predictions
        all_predictions = np.array(all_predictions)  # shape: (N_SEEDS, n_test)

        if is_classification:
            # For classification: average probabilities, then threshold
            ensemble_preds = all_predictions.mean(axis=0)
            print(f"\n  Ensemble (probability avg): mean={ensemble_preds.mean():.4f}")
        else:
            # For regression: simple mean
            ensemble_preds = all_predictions.mean(axis=0)
            print(f"\n  Ensemble (mean): mean={ensemble_preds.mean():.4f}")

        # Measure ensemble diversity
        pairwise_corr = np.corrcoef(all_predictions)
        avg_corr = (pairwise_corr.sum() - N_SEEDS) / (N_SEEDS * (N_SEEDS - 1))
        print(f"  Avg pairwise correlation: {avg_corr:.4f} (lower = more diverse = better)")

        task.record(fold_idx, ensemble_preds, params={
            "method": "multi_seed_ensemble",
            "n_seeds": N_SEEDS,
            "epochs": EPOCHS,
            "lr": LR,
        })
        print(f"  Fold {fold_idx} ensemble recorded.")

mb.validate()
print(f"\nEnsemble Scores:\n{mb.scores}")

results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
mb.to_file(f"{results_dir}/ensemble_{N_SEEDS}seed_{TASK_NAME}_results.json.gz")
print(f"Results saved.")
```

## Script 2: Cross-Architecture Ensemble

Train multiple different model architectures and combine their predictions. Weighted averaging based on validation performance often yields better results than any single model.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Cross-architecture ensemble: combine CGCNN + SchNet + RF predictions.
Weighted average based on per-fold validation performance.
Often the easiest path to competitive leaderboard scores.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import torch
from matbench.bench import MatbenchBenchmark
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TASK_NAME = "matbench_log_gvrh"  # Change as needed
CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS

mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

for task in mb.tasks:
    task.load()
    print(f"\nTask: {task.dataset_name}")

    for fold_idx in task.folds:
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx}")
        print(f"{'='*50}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        # Split training data for validation (to determine ensemble weights)
        val_size = min(0.1, 500 / len(train_inputs))
        tr_idx, val_idx = train_test_split(
            range(len(train_inputs)), test_size=val_size, random_state=42
        )

        model_predictions = {}  # {model_name: (val_preds, test_preds, val_score)}

        # --- Model 1: CGCNN ---
        print("\n  Training CGCNN...")
        # Replace with actual CGCNN training
        # cgcnn_model = CGCNN(config).to(device)
        # ... train ...
        # cgcnn_val_preds = cgcnn_model.predict(val_data)
        # cgcnn_test_preds = cgcnn_model.predict(test_data)
        cgcnn_val_preds = np.zeros(len(val_idx))    # placeholder
        cgcnn_test_preds = np.zeros(len(test_inputs))  # placeholder
        print("    CGCNN training complete.")

        # --- Model 2: SchNet ---
        print("  Training SchNet...")
        # Replace with actual SchNet training
        # schnet_model = SchNet(config).to(device)
        # ... train ...
        schnet_val_preds = np.zeros(len(val_idx))
        schnet_test_preds = np.zeros(len(test_inputs))
        print("    SchNet training complete.")

        # --- Model 3: Random Forest ---
        print("  Training Random Forest...")
        # RF needs featurized inputs (Magpie or structure features)
        # featurizer = matminer.featurizers...
        # X_train = featurizer.featurize_many(train_inputs)
        # For structure tasks, use structure fingerprints
        # For composition tasks, use Magpie features
        rf_val_preds = np.zeros(len(val_idx))
        rf_test_preds = np.zeros(len(test_inputs))
        print("    RF training complete.")

        # --- Compute validation scores and weights ---
        val_true = train_outputs.iloc[val_idx].values

        models = {
            "CGCNN": (cgcnn_val_preds, cgcnn_test_preds),
            "SchNet": (schnet_val_preds, schnet_test_preds),
            "RF": (rf_val_preds, rf_test_preds),
        }

        weights = {}
        for name, (val_p, test_p) in models.items():
            if is_classification:
                score = roc_auc_score(val_true, val_p)
                # Higher AUC = better = higher weight
                weights[name] = score
            else:
                score = mean_absolute_error(val_true, val_p)
                # Lower MAE = better = higher weight (use inverse)
                weights[name] = 1.0 / max(score, 1e-8)
            print(f"    {name} val score: {score:.4f}")

        # Normalize weights
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight
        print(f"\n  Ensemble weights: {weights}")

        # --- Weighted ensemble ---
        ensemble_preds = np.zeros(len(test_inputs))
        for name, (val_p, test_p) in models.items():
            ensemble_preds += weights[name] * test_p

        task.record(fold_idx, ensemble_preds, params={
            "method": "cross_architecture_ensemble",
            "models": list(models.keys()),
            "weights": weights,
        })
        print(f"  Fold {fold_idx} ensemble recorded.")

mb.validate()
print(f"\nCross-Architecture Ensemble Scores:\n{mb.scores}")

results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
mb.to_file(f"{results_dir}/cross_ensemble_{TASK_NAME}_results.json.gz")
print(f"Results saved.")
```

## Script 3: Transfer Learning

Pretrain on a large MatBench task (e.g., matbench_mp_e_form with 132K samples), then fine-tune on a small task. Especially useful for matbench_jdft2d (636 samples) and matbench_phonons (1,265 samples).

```python
#!/opt/conda/envs/matbench/bin/python
"""
Transfer learning: pretrain on large task, fine-tune on small task.
Strategy: train on matbench_mp_e_form, freeze early layers, fine-tune on target.
Best for: matbench_jdft2d, matbench_phonons, matbench_steels.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
PRETRAIN_TASK = "matbench_mp_e_form"  # Large source task (132K samples)
TARGET_TASK = "matbench_jdft2d"       # Small target task (636 samples)
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 100
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-5   # Much lower LR for fine-tuning
FROZEN_LAYERS = -2    # Freeze all but last 2 layers
BATCH_SIZE = 64

# === Phase 1: Pretrain on large task ===
print(f"Phase 1: Pretraining on {PRETRAIN_TASK}")
print(f"{'='*60}")

pretrain_mb = MatbenchBenchmark(autoload=False, subset=[PRETRAIN_TASK])
for pretrain_task in pretrain_mb.tasks:
    pretrain_task.load()

    # Use fold 0 for pretraining (we just need the weights, not recording)
    train_inputs, train_outputs = pretrain_task.get_train_and_val_data(0)
    print(f"  Pretrain data: {len(train_inputs)} samples")

    # ========================================
    # YOUR MODEL INITIALIZATION HERE
    # model = YourGNN(config).to(device)
    # ========================================

    # Pretrain loop (replace with actual training)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR)
    # for epoch in range(PRETRAIN_EPOCHS):
    #     model.train()
    #     for batch in pretrain_loader:
    #         optimizer.zero_grad()
    #         pred = model(batch)
    #         loss = criterion(pred, batch.y)
    #         loss.backward()
    #         optimizer.step()
    #     if (epoch + 1) % 50 == 0:
    #         print(f"  Pretrain epoch {epoch+1}/{PRETRAIN_EPOCHS}")

    # Save pretrained weights
    pretrain_dir = f"/workspace/group/matbench/models/transfer/{PRETRAIN_TASK}"
    os.makedirs(pretrain_dir, exist_ok=True)
    # torch.save(model.state_dict(), f"{pretrain_dir}/pretrained.pt")
    print(f"  Pretrained model saved to {pretrain_dir}/pretrained.pt")

# === Phase 2: Fine-tune on small task ===
print(f"\nPhase 2: Fine-tuning on {TARGET_TASK}")
print(f"{'='*60}")

target_mb = MatbenchBenchmark(autoload=False, subset=[TARGET_TASK])

for task in target_mb.tasks:
    task.load()
    print(f"\nTarget task: {task.dataset_name} ({len(task.df)} samples)")

    for fold_idx in task.folds:
        print(f"\n  --- Fold {fold_idx} ---")
        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)
        print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

        # Load pretrained model
        # model = YourGNN(config).to(device)
        # model.load_state_dict(torch.load(f"{pretrain_dir}/pretrained.pt"))

        # Freeze all but last N layers
        # all_params = list(model.named_parameters())
        # n_total = len(all_params)
        # freeze_up_to = n_total + FROZEN_LAYERS  # e.g., -2 means freeze all but last 2
        # for i, (name, param) in enumerate(all_params):
        #     if i < freeze_up_to:
        #         param.requires_grad = False
        #         print(f"    Frozen: {name}")
        #     else:
        #         param.requires_grad = True
        #         print(f"    Trainable: {name}")

        # Fine-tune with low learning rate
        # trainable_params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = torch.optim.AdamW(trainable_params, lr=FINETUNE_LR)

        # for epoch in range(FINETUNE_EPOCHS):
        #     model.train()
        #     for batch in finetune_loader:
        #         optimizer.zero_grad()
        #         pred = model(batch)
        #         loss = criterion(pred, batch.y)
        #         loss.backward()
        #         optimizer.step()
        #     if (epoch + 1) % 25 == 0:
        #         print(f"    Fine-tune epoch {epoch+1}/{FINETUNE_EPOCHS}")

        # Predict
        # model.eval()
        # with torch.no_grad():
        #     predictions = model.predict(test_data)

        predictions = np.zeros(len(test_inputs))  # Replace with actual

        task.record(fold_idx, predictions, params={
            "method": "transfer_learning",
            "pretrain_task": PRETRAIN_TASK,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "finetune_lr": FINETUNE_LR,
            "frozen_layers": FROZEN_LAYERS,
        })
        print(f"  Fold {fold_idx} fine-tuned and recorded.")

        # Save fine-tuned model
        ft_dir = f"/workspace/group/matbench/models/transfer/{TARGET_TASK}"
        os.makedirs(ft_dir, exist_ok=True)
        # torch.save(model.state_dict(), f"{ft_dir}/fold_{fold_idx}.pt")

target_mb.validate()
print(f"\nTransfer Learning Scores:\n{target_mb.scores}")

results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
target_mb.to_file(f"{results_dir}/transfer_{TARGET_TASK}_results.json.gz")
print(f"Results saved.")
```

## Script 4: Hyperparameter Optimization

Systematic random search over hyperparameter space. Run abbreviated training (100 epochs) per configuration, select the best, then retrain with full epochs.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Hyperparameter optimization via random search.
Strategy: run short training (100 epochs) per config, rank, retrain best.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import json
import time
import numpy as np
import torch
from matbench.bench import MatbenchBenchmark
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
TASK_NAME = "matbench_mp_e_form"  # Change as needed
SEARCH_BUDGET = 20                # Number of random configs to try
SHORT_EPOCHS = 100                # Abbreviated training for search
FULL_EPOCHS = 500                 # Full training for best config

# Define search space
SEARCH_SPACE = {
    "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
    "hidden_dim": {"type": "choice", "values": [64, 128, 256, 512]},
    "num_layers": {"type": "choice", "values": [2, 3, 4, 5, 6]},
    "batch_size": {"type": "choice", "values": [32, 64, 128, 256]},
    "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
    "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
}

def sample_config(space):
    """Sample a random configuration from the search space."""
    config = {}
    for key, spec in space.items():
        if spec["type"] == "choice":
            config[key] = np.random.choice(spec["values"])
        elif spec["type"] == "log_uniform":
            log_val = np.random.uniform(np.log(spec["low"]), np.log(spec["high"]))
            config[key] = float(np.exp(log_val))
        elif spec["type"] == "uniform":
            config[key] = float(np.random.uniform(spec["low"], spec["high"]))
    return config

# Phase 1: Random search with abbreviated training
print(f"Phase 1: Random Search ({SEARCH_BUDGET} configs, {SHORT_EPOCHS} epochs each)")
print(f"{'='*60}")

mb_search = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
for task in mb_search.tasks:
    task.load()

    # Use fold 0 for hyperparameter search
    train_inputs, train_outputs = task.get_train_and_val_data(0)

    # Split into search-train and search-val
    val_size = min(0.15, 2000 / len(train_inputs))
    tr_idx, val_idx = train_test_split(
        range(len(train_inputs)), test_size=val_size, random_state=42
    )
    print(f"  Search train: {len(tr_idx)}, Search val: {len(val_idx)}")

    results = []

    for trial in range(SEARCH_BUDGET):
        config = sample_config(SEARCH_SPACE)
        print(f"\n  Trial {trial+1}/{SEARCH_BUDGET}: {config}")

        start_time = time.time()

        # ========================================
        # YOUR MODEL TRAINING WITH config HERE
        # model = YourGNN(
        #     hidden_dim=int(config["hidden_dim"]),
        #     num_layers=int(config["num_layers"]),
        #     dropout=config["dropout"],
        # ).to(device)
        # optimizer = torch.optim.AdamW(
        #     model.parameters(),
        #     lr=config["learning_rate"],
        #     weight_decay=config["weight_decay"],
        # )
        #
        # for epoch in range(SHORT_EPOCHS):
        #     model.train()
        #     ... (train on tr_idx subset)
        #
        # model.eval()
        # val_preds = model.predict(val_data)
        # val_score = mean_absolute_error(val_true, val_preds)
        # ========================================

        val_score = np.random.uniform(0.01, 0.1)  # Placeholder
        elapsed = time.time() - start_time

        result = {
            "trial": trial,
            "config": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in config.items()},
            "val_score": float(val_score),
            "time_seconds": elapsed,
        }
        results.append(result)
        print(f"    Val score: {val_score:.6f} ({elapsed:.1f}s)")

    # Sort by validation score (lower MAE = better for regression)
    results.sort(key=lambda x: x["val_score"])

    print(f"\n{'='*60}")
    print("Top 5 configurations:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. Score={r['val_score']:.6f} | Config: {r['config']}")

    # Save all search results
    hpopt_dir = "/workspace/group/matbench/models/hpopt"
    os.makedirs(hpopt_dir, exist_ok=True)
    with open(f"{hpopt_dir}/{TASK_NAME}_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSearch results saved to {hpopt_dir}/{TASK_NAME}_search_results.json")

    # Phase 2: Retrain best config with full epochs
    best_config = results[0]["config"]
    print(f"\nPhase 2: Retraining best config with {FULL_EPOCHS} epochs")
    print(f"Best config: {best_config}")
    print(f"{'='*60}")

# Full benchmark with best config
best_mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
for task in best_mb.tasks:
    task.load()

    for fold_idx in task.folds:
        print(f"\n  --- Fold {fold_idx} (full training) ---")
        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        # ========================================
        # TRAIN WITH best_config AND FULL_EPOCHS
        # model = YourGNN(**best_config).to(device)
        # ... full training loop ...
        # predictions = model.predict(test_data)
        # ========================================

        predictions = np.zeros(len(test_inputs))  # Replace
        task.record(fold_idx, predictions, params={
            "method": "hpopt_best",
            **best_config,
            "epochs": FULL_EPOCHS,
        })
        print(f"  Fold {fold_idx} done.")

best_mb.validate()
print(f"\nOptimized Scores:\n{best_mb.scores}")

results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
best_mb.to_file(f"{results_dir}/hpopt_{TASK_NAME}_results.json.gz")
print(f"Results saved.")
```

## Script 5: Feature Engineering

Combine GNN learned embeddings with handcrafted features from matminer for a hybrid approach that often outperforms either alone.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Hybrid feature engineering: combine GNN embeddings + matminer handcrafted features.
Strategy: extract penultimate-layer embeddings from trained GNN, concatenate with
structural/compositional features, train a final prediction layer.
"""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TASK_NAME = "matbench_log_gvrh"  # Change as needed
CLASSIFICATION_TASKS = {"matbench_expt_is_metal", "matbench_glass", "matbench_mp_is_metal"}
is_classification = TASK_NAME in CLASSIFICATION_TASKS

mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])

for task in mb.tasks:
    task.load()
    print(f"\nTask: {task.dataset_name}")

    for fold_idx in task.folds:
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx}")
        print(f"{'='*50}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs = task.get_test_data(fold_idx, include_target=False)

        # === Part A: GNN Embeddings ===
        print("  Extracting GNN embeddings...")

        # Load or train a GNN model (CGCNN, SchNet, etc.)
        # model = load_trained_gnn(fold_idx)

        # Extract penultimate layer embeddings
        # def get_embeddings(model, structures):
        #     model.eval()
        #     embeddings = []
        #     with torch.no_grad():
        #         for struct in structures:
        #             graph = structure_to_graph(struct)
        #             emb = model.get_embedding(graph)  # Hook into penultimate layer
        #             embeddings.append(emb.cpu().numpy())
        #     return np.array(embeddings)
        #
        # train_gnn_emb = get_embeddings(model, train_inputs)
        # test_gnn_emb = get_embeddings(model, test_inputs)

        # Placeholder: replace with actual embeddings
        EMB_DIM = 128
        train_gnn_emb = np.random.randn(len(train_inputs), EMB_DIM)
        test_gnn_emb = np.random.randn(len(test_inputs), EMB_DIM)
        print(f"    GNN embeddings: {train_gnn_emb.shape}")

        # === Part B: Handcrafted Features ===
        print("  Computing matminer features...")

        def compute_structure_features(structures):
            """Compute handcrafted features for crystal structures."""
            features = []
            for i, struct in enumerate(structures):
                try:
                    feat = {
                        "density": struct.density,
                        "volume": struct.volume,
                        "volume_per_atom": struct.volume / len(struct),
                        "num_sites": len(struct),
                        "avg_atomic_number": np.mean([s.specie.Z for s in struct]),
                        "avg_atomic_mass": np.mean([s.specie.atomic_mass for s in struct]),
                        "avg_electronegativity": np.mean([
                            s.specie.X for s in struct if hasattr(s.specie, 'X') and s.specie.X is not None
                        ]) if any(hasattr(s.specie, 'X') and s.specie.X for s in struct) else 0,
                    }

                    # Lattice features
                    lattice = struct.lattice
                    feat["a"] = lattice.a
                    feat["b"] = lattice.b
                    feat["c"] = lattice.c
                    feat["alpha"] = lattice.alpha
                    feat["beta"] = lattice.beta
                    feat["gamma"] = lattice.gamma

                    features.append(feat)
                except Exception as e:
                    features.append({k: 0.0 for k in [
                        "density", "volume", "volume_per_atom", "num_sites",
                        "avg_atomic_number", "avg_atomic_mass", "avg_electronegativity",
                        "a", "b", "c", "alpha", "beta", "gamma"
                    ]})

                if (i + 1) % 1000 == 0:
                    print(f"      Featurized {i+1}/{len(structures)}")

            return pd.DataFrame(features).values

        train_struct_feat = compute_structure_features(train_inputs)
        test_struct_feat = compute_structure_features(test_inputs)
        print(f"    Structure features: {train_struct_feat.shape}")

        # === Part C: Concatenate and Scale ===
        print("  Concatenating features...")
        train_combined = np.hstack([train_gnn_emb, train_struct_feat])
        test_combined = np.hstack([test_gnn_emb, test_struct_feat])
        print(f"    Combined features: {train_combined.shape}")

        # Handle NaN/Inf
        train_combined = np.nan_to_num(train_combined, nan=0.0, posinf=0.0, neginf=0.0)
        test_combined = np.nan_to_num(test_combined, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_combined)
        test_scaled = scaler.transform(test_combined)

        # === Part D: Train Final Predictor ===
        print("  Training final predictor (GBR on combined features)...")
        if is_classification:
            final_model = GradientBoostingClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        else:
            final_model = GradientBoostingRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=42
            )

        final_model.fit(train_scaled, train_outputs.values)

        if is_classification:
            predictions = final_model.predict_proba(test_scaled)[:, 1]
        else:
            predictions = final_model.predict(test_scaled)

        task.record(fold_idx, predictions, params={
            "method": "hybrid_gnn_features",
            "gnn_emb_dim": EMB_DIM,
            "struct_feat_dim": train_struct_feat.shape[1],
            "final_model": "GradientBoosting",
        })
        print(f"  Fold {fold_idx} recorded. Predictions: mean={predictions.mean():.4f}")

mb.validate()
print(f"\nHybrid Feature Engineering Scores:\n{mb.scores}")

results_dir = "/workspace/group/matbench/results"
os.makedirs(results_dir, exist_ok=True)
mb.to_file(f"{results_dir}/hybrid_{TASK_NAME}_results.json.gz")
print(f"Results saved.")
```

## Section 6: Complete SOTA Strategy Guide

A phased roadmap for systematically achieving top MatBench leaderboard positions.

### Phase 1: Establish Baseline (Day 1)

**Goal:** Get scores for all 13 tasks with simple models.

```
1. Composition tasks (5 tasks):
   - Use composition-models/ skill: RF + Magpie features
   - Run: matbench_steels, matbench_expt_gap, matbench_dielectric,
          matbench_expt_is_metal, matbench_glass
   - Expected: ~30 min total

2. Structure tasks (8 tasks):
   - Use structure-gnn/ skill: CGCNN with default config
   - Run: all 8 structure tasks
   - Expected: ~2-4 hours for small tasks, ~8-12 hours for mp_e_form

3. Record all results using evaluation-submission/ Script 1
4. Compare to leaderboard using evaluation-submission/ Script 3
5. Identify: which tasks are you closest to SOTA?
```

### Phase 2: Reproduce SOTA (Day 2-3)

**Goal:** Run known SOTA models to verify you can match published results.

```
1. Install and run ALIGNN on all structure tasks
   - See Section 1 of this skill
   - Focus on: matbench_perovskites, matbench_mp_e_form, matbench_mp_gap

2. Install and run MODNet on all tasks
   - See Section 2 of this skill
   - MODNet is especially strong on: matbench_dielectric, matbench_glass, matbench_jdft2d

3. Compare ALIGNN vs CGCNN vs MODNet per task
4. Use evaluation-submission/ Script 3 for comparison charts
```

### Phase 3: Hyperparameter Sweep (Day 3-5)

**Goal:** Find optimal configurations for your best model per task.

```
1. For each task, take the best model from Phase 2
2. Run Script 4 (random search) with SEARCH_BUDGET=20
3. Focus on tasks where you are closest to SOTA (highest impact)
4. Key hyperparameters to sweep:
   - learning_rate: 1e-5 to 1e-2 (log scale)
   - hidden_dim: 64 to 512
   - num_layers: 2 to 6
   - batch_size: 32 to 256
5. Use AMP (automatic mixed precision) for ~2x training speed:
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       pred = model(batch)
       loss = criterion(pred, batch.y)
```

### Phase 4: Ensemble (Day 5-6)

**Goal:** Combine models for better accuracy. This alone often reaches top-5.

```
1. Multi-seed ensemble (Script 1):
   - Take best config from Phase 3
   - Train with 5 different seeds
   - Average predictions
   - Typical improvement: 5-15% MAE reduction

2. Cross-architecture ensemble (Script 2):
   - Combine top 2-3 model types per task
   - Weight by validation performance
   - Typical improvement: 10-20% over best single model

3. Record all ensemble results
4. Compare to leaderboard -- you should be competitive now
```

### Phase 5: Task-Specific Optimization (Day 7+)

**Goal:** Targeted improvements for remaining gaps.

```
1. Small tasks (steels, jdft2d, phonons):
   - Transfer learning from mp_e_form (Script 3)
   - Data augmentation (perturbation of atomic positions)
   - Feature engineering (Script 5)

2. Composition tasks (expt_gap, dielectric, expt_is_metal, glass):
   - Enhanced Magpie + extra features (oxidation states, ionic radii)
   - MODNet with larger ensemble (n_models=10)
   - Try CrabNet or Roost if available

3. Large structure tasks (mp_gap, mp_e_form, mp_is_metal):
   - Deeper GNNs (6-8 layers with residual connections)
   - DimeNet++ or PaiNN for angular information
   - Larger batch + longer training with cosine annealing

4. Classification tasks (is_metal, glass):
   - Threshold tuning on validation set
   - Balanced loss weighting for class imbalance
   - Calibration (temperature scaling)
```

## Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| n_ensemble | Number of models in ensemble | 3-10 |
| ensemble_method | "mean" (regression) or "probability_mean" (classification) | -- |
| frozen_layers | Layers to freeze in transfer learning (from end) | -2 to -4 |
| fine_tune_lr | Learning rate for fine-tuning | 1e-6 to 1e-4 |
| search_budget | Number of HP configs to try | 10-50 |
| noise_std | Perturbation noise for augmentation | 0.01-0.05 |

## Common Issues

| Issue | Solution |
|-------|---------|
| Ensemble worse than best single model | Models too correlated; increase diversity (different seeds, architectures, or hyperparameters) |
| Transfer learning degrades performance | Source and target tasks too dissimilar; try unfreezing more layers or using a smaller fine-tune LR |
| Feature noise from handcrafted features | Use feature selection (mutual information, importance ranking) to remove noisy features |
| GPU OOM for ensemble training | Train models sequentially, not in parallel; save checkpoints and load for prediction only |
| Metric plateaus near noise floor | Some tasks have inherent noise; ensemble of 5+ diverse models is the best strategy at this point |
| HP search too slow | Use shorter SHORT_EPOCHS (50-100); focus search on most impactful params (LR, hidden_dim) |
| Validation score doesn't match test | Ensure validation split is representative; use stratified splits for classification tasks |
