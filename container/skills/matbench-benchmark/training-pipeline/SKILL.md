# Train MatBench Models with GPU and Advanced Scheduling

## A100 GPU Optimization (MANDATORY — READ FIRST)

**The A100-SXM4-80GB has 80 GB VRAM but previous runs only used 3-6 GB (< 8%). This section fixes that.**

### Root Causes of Low GPU Utilization

| Problem | Impact | Found In Previous Runs |
|---------|--------|----------------------|
| **batch_size=32 on A100** | GPU idle 90%+ of the time waiting for tiny batches | CGCNN: jdft2d=32, phonons=32 |
| **num_workers=0** | Data loading runs on main thread, GPU waits for CPU | All DataLoaders |
| **pin_memory=False** | CPU→GPU memory transfer not optimized | All DataLoaders |
| **No data prefetching** | GPU idle during data preparation | All scripts |
| **Small models (256 hidden)** | Model doesn't saturate GPU compute | All models |

### Mandatory DataLoader Configuration for A100

**EVERY DataLoader in EVERY training script MUST use these settings:**

```python
# ❌ WRONG — wastes 90% of A100
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

# ✅ CORRECT — full A100 utilization
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,      # See batch size table below
    shuffle=True,
    num_workers=4,              # 4 parallel data loading workers
    pin_memory=True,            # Pin CPU memory for fast GPU transfer
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=2,          # Pre-load 2 batches ahead
)
```

### Recommended Batch Sizes for A100 80GB

| Dataset Size | Model Type | Recommended batch_size | Expected VRAM Usage |
|-------------|-----------|----------------------|-------------------|
| < 1,000 | GNN (CGCNN, SchNet) | 128-256 | 10-20 GB |
| 1,000 - 10,000 | GNN | 256-512 | 15-30 GB |
| 10,000 - 50,000 | GNN | 512-1024 | 20-40 GB |
| > 50,000 | GNN | 1024-2048 | 30-60 GB |
| Any | MLP on features | 2048-8192 | 5-15 GB |
| Any | MLP on features (AMP) | 4096-16384 | 5-15 GB |

**Rule of thumb**: If `nvidia-smi` shows < 20 GB VRAM used during training, **double the batch size**.

### Mandatory A100 Optimizations

Every training script MUST include ALL of these:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- A100 optimizations (MANDATORY) ---
if device.type == "cuda":
    # 1. Enable TF32 for ~3x faster matmul (A100 exclusive)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 2. Enable cuDNN autotuner (finds fastest conv algorithms)
    torch.backends.cudnn.benchmark = True

    # 3. Print GPU info for verification
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_mem / 1e9:.1f} GB")
    print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
# --- End A100 optimizations ---
```

### Mixed Precision (AMP) — Use ALWAYS on A100

AMP gives ~1.5-2x speedup with negligible accuracy loss on A100:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    batch = batch.to(device)
    optimizer.zero_grad()

    with autocast(device_type="cuda"):    # FP16 forward pass
        pred = model(batch)
        loss = criterion(pred, target)

    scaler.scale(loss).backward()         # Scaled FP16 backward
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    scaler.step(optimizer)
    scaler.update()
```

### GPU Monitoring During Training

Add this to your training loop to track utilization:

```python
# Print GPU stats every N epochs
if epoch % 10 == 0 and device.type == "cuda":
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  GPU Memory: {mem_used:.1f} / {mem_total:.1f} GB ({mem_used/mem_total*100:.0f}%)")
```

If you see < 20% VRAM usage, **immediately increase batch_size**.

### Complete Optimized DataLoader Example

```python
def make_loaders(train_data, test_data, batch_size, is_graph=True):
    """Create optimized DataLoaders for A100."""
    common = dict(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    if is_graph:
        from torch_geometric.loader import DataLoader as PyGLoader
        train_loader = PyGLoader(train_data, batch_size=batch_size, shuffle=True, **common)
        test_loader = PyGLoader(test_data, batch_size=batch_size * 2, shuffle=False, **common)
    else:
        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **common)
        test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, test_loader
```

---

## When to Use

- You have a model and data ready and need an efficient PyTorch training loop
- You need learning rate scheduling (cosine, plateau, one-cycle, warmup)
- You want early stopping with best-model checkpointing
- You want TensorBoard logging for experiment tracking
- You need hyperparameter tuning over a search space
- You want to maximize A100 GPU utilization with mixed precision (AMP)

## Method Selection

```
Start here: What training aspect do you need?
│
├─ Need a basic training loop with GPU support?
│  └─ → Standard GPU Training Loop (Script 1)
│     Complete train/val loop with CUDA, gradient clipping, checkpointing
│
├─ Want to tune the learning rate schedule?
│  └─ → Learning Rate Schedulers (Script 2)
│     CosineAnnealing, ReduceLROnPlateau, OneCycleLR, Warmup+Cosine
│
├─ Need to stop training when no improvement?
│  └─ → Early Stopping (Script 3)
│     Monitors val loss, saves best model, restores on stop
│
├─ Want to search over hyperparameters?
│  └─ → Hyperparameter Search (Script 4)
│     Random search over LR, hidden, layers, dropout, batch size
│
├─ Want to visualize training progress?
│  └─ → TensorBoard Integration (Script 5)
│     Log losses, metrics, LR per epoch. View in TensorBoard.
│
└─ Want maximum GPU speed on A100?
   └─ → Mixed Precision / AMP (Script 6)
      FP16 autocast + GradScaler for ~2x speedup
```

## Prerequisites

- **Python:** `/opt/conda/envs/matbench/bin/python`
- **Packages:** torch (CUDA 12.8), tensorboard, scikit-learn, numpy
- **GPU:** NVIDIA A100-SXM4-80GB (80 GB VRAM, CUDA 12.8)
- **Storage:** `/workspace/group/matbench/` (data disk)

## Detailed Steps

---

### Script 1: Standard GPU Training Loop

A complete, reusable training function for any PyTorch model on MatBench tasks.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Standard GPU Training Loop for MatBench.
Works with any PyTorch model (GNN, MLP, etc.).
Includes: CUDA auto-detect, gradient clipping, checkpointing, per-epoch logging.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ── Training function ─────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, epochs=300, lr=1e-3,
                weight_decay=1e-5, grad_clip=10.0, task_type="regression",
                save_dir="/workspace/group/matbench/models",
                model_name="model"):
    """Complete training loop with validation, checkpointing, and logging.

    Args:
        model: PyTorch model (must accept batch from DataLoader)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch.device ("cuda" or "cpu")
        epochs: Max training epochs
        lr: Initial learning rate
        weight_decay: L2 regularization
        grad_clip: Max gradient norm (0 to disable)
        task_type: "regression" (MSELoss) or "classification" (BCEWithLogitsLoss)
        save_dir: Directory for model checkpoints
        model_name: Name prefix for checkpoint file

    Returns:
        dict with best_state_dict, best_val_metric, train_losses, val_metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # Loss function
    if task_type == "regression":
        criterion = nn.MSELoss()
    elif task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Tracking
    train_losses = []
    val_metrics = []
    best_val_metric = float("inf")
    best_state = None
    ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")

    print(f"\nTraining {model_name}")
    print(f"  Epochs: {epochs}, LR: {lr}, Weight decay: {weight_decay}")
    print(f"  Task type: {task_type}, Grad clip: {grad_clip}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )

            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
            n_train += batch.num_graphs

        train_loss = epoch_loss / n_train
        train_losses.append(train_loss)

        # ── Validate ──
        model.eval()
        preds_all, trues_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                preds_all.append(pred.cpu().numpy())
                trues_all.append(batch.y.cpu().numpy())

        preds_all = np.concatenate(preds_all)
        trues_all = np.concatenate(trues_all)

        if task_type == "regression":
            val_metric = np.mean(np.abs(preds_all - trues_all))  # MAE
            metric_name = "MAE"
        else:
            from sklearn.metrics import roc_auc_score
            probs = 1.0 / (1.0 + np.exp(-preds_all))
            val_metric = 1.0 - roc_auc_score(trues_all, probs)  # 1-AUC (lower=better)
            metric_name = "1-AUC"

        val_metrics.append(val_metric)
        scheduler.step()
        elapsed = time.time() - t0

        # ── Logging ──
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss={train_loss:.5f} | "
                  f"Val {metric_name}={val_metric:.5f} | "
                  f"LR={current_lr:.2e} | {elapsed:.1f}s")

        # ── Checkpoint ──
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

    print(f"\nBest val {metric_name}: {best_val_metric:.5f}")
    print(f"Checkpoint: {ckpt_path}")

    return {
        "best_state_dict": best_state,
        "best_val_metric": best_val_metric,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "ckpt_path": ckpt_path,
    }


# ── Plot training curves ─────────────────────────────────────────────────────
def plot_training_curves(train_losses, val_metrics, model_name="model",
                         metric_name="MAE",
                         save_dir="/workspace/group/matbench/plots"):
    """Save training loss and validation metric curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (MSE)")
    ax1.set_title(f"{model_name} - Training Loss")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_metrics, linewidth=1.5, color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(f"Validation {metric_name}")
    ax2.set_title(f"{model_name} - Validation {metric_name}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")
    return path


# ── Demo usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Standard GPU Training Loop - Demo")
    print("This module provides train_model() and plot_training_curves().")
    print()
    print("Usage:")
    print("  from training_loop import train_model, plot_training_curves")
    print()
    print("  result = train_model(")
    print("      model=my_model,")
    print("      train_loader=train_loader,")
    print("      val_loader=val_loader,")
    print("      device=device,")
    print("      epochs=300,")
    print("      lr=1e-3,")
    print("      model_name='cgcnn_mp_e_form_fold0',")
    print("  )")
    print()
    print("  plot_training_curves(")
    print("      result['train_losses'],")
    print("      result['val_metrics'],")
    print("      model_name='cgcnn_mp_e_form_fold0',")
    print("  )")
    print("\nDone.")
```

---

### Script 2: Learning Rate Schedulers

Compare and use different LR scheduling strategies for MatBench training.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Learning Rate Schedulers for MatBench.
Demonstrates CosineAnnealing, ReduceLROnPlateau, OneCycleLR, and Warmup+Cosine.
"""
import os, math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 1. CosineAnnealingLR (recommended for MatBench GNNs) ─────────────────────
def setup_cosine_annealing(optimizer, epochs, eta_min=1e-7):
    """Cosine annealing from initial LR to eta_min.

    Best for: GNN training on MatBench (CGCNN, SchNet, DimeNet++).
    Smoothly decays LR, avoids sudden drops.

    Usage:
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = setup_cosine_annealing(optimizer, epochs=300)
        # In training loop: scheduler.step() after each epoch
    """
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


# ── 2. ReduceLROnPlateau ─────────────────────────────────────────────────────
def setup_reduce_on_plateau(optimizer, factor=0.5, patience=20, min_lr=1e-7):
    """Reduce LR when validation loss stops improving.

    Best for: When you do not know how many epochs are needed.
    Adaptive — only reduces LR when progress stalls.

    Usage:
        scheduler = setup_reduce_on_plateau(optimizer)
        # In training loop: scheduler.step(val_loss) after each epoch
    """
    return ReduceLROnPlateau(
        optimizer, mode="min", factor=factor,
        patience=patience, min_lr=min_lr, verbose=True
    )


# ── 3. OneCycleLR (fast convergence) ─────────────────────────────────────────
def setup_one_cycle(optimizer, max_lr, epochs, steps_per_epoch):
    """One-cycle policy: ramp up then anneal down.

    Best for: Fast convergence, fewer epochs needed.
    Can reach good performance in 100-150 epochs instead of 300.

    Usage:
        scheduler = setup_one_cycle(optimizer, max_lr=1e-3,
                                     epochs=150, steps_per_epoch=len(train_loader))
        # In training loop: scheduler.step() after each BATCH (not epoch)
    """
    return OneCycleLR(
        optimizer, max_lr=max_lr,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.3,       # 30% warmup
        anneal_strategy="cos",
        div_factor=25.0,     # initial_lr = max_lr / 25
        final_div_factor=1e4 # final_lr = initial_lr / 10000
    )


# ── 4. Warmup + Cosine Decay (custom) ────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay.

    Best for: Large models or large learning rates that need gentle start.
    Standard in transformer-based models, also helps GNNs.

    Usage:
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=20,
                                           total_epochs=300, min_lr=1e-7)
        # In training loop: scheduler.step() after each epoch
    """

    def __init__(self, optimizer, warmup_epochs=20, total_epochs=300,
                 min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = base_lr * (self.current_epoch / self.warmup_epochs)
            else:
                # Cosine decay
                progress = (self.current_epoch - self.warmup_epochs) / (
                    self.total_epochs - self.warmup_epochs
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ── Visualization: compare all schedulers ─────────────────────────────────────
if __name__ == "__main__":
    EPOCHS = 300
    LR = 1e-3
    SAVE_DIR = "/workspace/group/matbench/plots"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Dummy model for optimizer creation
    dummy = nn.Linear(10, 1)

    schedulers = {}

    # 1. Cosine Annealing
    opt1 = torch.optim.Adam(dummy.parameters(), lr=LR)
    sched1 = setup_cosine_annealing(opt1, EPOCHS)
    lrs1 = []
    for _ in range(EPOCHS):
        lrs1.append(opt1.param_groups[0]["lr"])
        sched1.step()
    schedulers["CosineAnnealing"] = lrs1

    # 2. ReduceLROnPlateau (simulate plateaus)
    opt2 = torch.optim.Adam(dummy.parameters(), lr=LR)
    sched2 = setup_reduce_on_plateau(opt2, factor=0.5, patience=20)
    lrs2 = []
    for e in range(EPOCHS):
        lrs2.append(opt2.param_groups[0]["lr"])
        # Simulate: loss improves for 50 epochs, then plateaus periodically
        fake_loss = 1.0 / (e + 1) if e % 70 < 50 else 0.02
        sched2.step(fake_loss)
    schedulers["ReduceOnPlateau"] = lrs2

    # 3. OneCycleLR
    opt3 = torch.optim.Adam(dummy.parameters(), lr=LR / 25)
    sched3 = setup_one_cycle(opt3, max_lr=LR, epochs=EPOCHS,
                              steps_per_epoch=100)
    lrs3 = []
    for _ in range(EPOCHS):
        lr_epoch = opt3.param_groups[0]["lr"]
        lrs3.append(lr_epoch)
        for _ in range(100):  # simulate batches
            sched3.step()
    schedulers["OneCycleLR"] = lrs3

    # 4. Warmup + Cosine
    opt4 = torch.optim.Adam(dummy.parameters(), lr=LR)
    sched4 = WarmupCosineScheduler(opt4, warmup_epochs=20,
                                    total_epochs=EPOCHS)
    lrs4 = []
    for _ in range(EPOCHS):
        lrs4.append(opt4.param_groups[0]["lr"])
        sched4.step()
    schedulers["Warmup+Cosine"] = lrs4

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    for (name, lrs), color in zip(schedulers.items(), colors):
        ax.plot(lrs, label=name, linewidth=2, color=color)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("LR Scheduler Comparison (initial LR=1e-3)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, "lr_scheduler_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scheduler comparison plot saved: {path}")

    # Print recommendations
    print("\nRecommendations for MatBench:")
    print("  CGCNN:     CosineAnnealing (lr=1e-3, 300 epochs)")
    print("  SchNet:    Warmup+Cosine (lr=5e-4, warmup=20, 300 epochs)")
    print("  DimeNet++: Warmup+Cosine (lr=1e-4, warmup=30, 300 epochs)")
    print("  Quick run: OneCycleLR (lr=1e-3, 150 epochs)")
    print("\nDone.")
```

---

### Script 3: Early Stopping

Reusable early stopping class with best-model checkpointing.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Early Stopping for MatBench Training.
Monitors validation metric, saves best model, stops when no improvement.
"""
import os, time, copy
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping handler with best-model checkpointing.

    Monitors a validation metric (lower is better by default).
    Saves the best model checkpoint and restores it when training stops.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        save_path: Path to save best model checkpoint.
        mode: "min" (lower is better) or "max" (higher is better).
        verbose: Print messages when saving or stopping.
    """

    def __init__(self, patience=50, min_delta=0.0,
                 save_path="/workspace/group/matbench/models/checkpoint.pt",
                 mode="min", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, metric, model, epoch=0):
        """Check if training should stop.

        Args:
            metric: Current validation metric value.
            model: PyTorch model to checkpoint.
            epoch: Current epoch number (for logging).

        Returns:
            True if training should stop.
        """
        if self.mode == "min":
            score = -metric
            improved = (self.best_score is None or
                        score > self.best_score + self.min_delta)
        else:
            score = metric
            improved = (self.best_score is None or
                        score > self.best_score + self.min_delta)

        if improved:
            if self.verbose and self.best_score is not None:
                prev = -self.best_score if self.mode == "min" else self.best_score
                print(f"  EarlyStopping: metric improved "
                      f"{prev:.5f} -> {metric:.5f}, saving model")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f"  EarlyStopping: no improvement for "
                      f"{self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: stopping at epoch {epoch} "
                          f"(best was epoch {self.best_epoch})")

        return self.should_stop

    def load_best_model(self, model):
        """Load the best model checkpoint."""
        state = torch.load(self.save_path, weights_only=False)
        model.load_state_dict(state)
        if self.verbose:
            print(f"  Loaded best model from epoch {self.best_epoch}")
        return model


# ── Example: training loop with EarlyStopping ────────────────────────────────
def train_with_early_stopping(model, train_loader, val_loader, device,
                              epochs=300, lr=1e-3, patience=50,
                              save_path="/workspace/group/matbench/models/model_best.pt"):
    """Complete training loop with integrated early stopping.

    Returns:
        dict with best_epoch, best_mae, train_losses, val_maes
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.MSELoss()

    early_stop = EarlyStopping(
        patience=patience,
        save_path=save_path,
        mode="min",
        verbose=True,
    )

    train_losses = []
    val_maes = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
        train_loss = total_loss / n
        train_losses.append(train_loss)

        # Validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                trues.append(batch.y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_mae = np.mean(np.abs(preds - trues))
        val_maes.append(val_mae)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss={train_loss:.5f} | MAE={val_mae:.5f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping check
        if early_stop(val_mae, model, epoch):
            break

    # Restore best model
    model = early_stop.load_best_model(model)

    return {
        "best_epoch": early_stop.best_epoch,
        "best_mae": -early_stop.best_score,
        "train_losses": train_losses,
        "val_maes": val_maes,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("EarlyStopping - Demo")
    print()
    print("Usage:")
    print("  early_stop = EarlyStopping(patience=50, save_path='best.pt')")
    print()
    print("  for epoch in range(1, epochs + 1):")
    print("      train_loss = train_one_epoch(...)")
    print("      val_mae = evaluate(...)")
    print("      if early_stop(val_mae, model, epoch):")
    print("          break")
    print()
    print("  model = early_stop.load_best_model(model)")
    print()

    # Simulate a training run
    print("Simulating training with early stopping (patience=10):")
    dummy_model = nn.Linear(10, 1)
    es = EarlyStopping(
        patience=10,
        save_path="/workspace/group/matbench/models/_demo_checkpoint.pt",
        verbose=True,
    )

    np.random.seed(42)
    for epoch in range(1, 101):
        # Simulate: metric improves then plateaus
        if epoch < 30:
            metric = 1.0 / epoch + np.random.normal(0, 0.01)
        else:
            metric = 0.035 + np.random.normal(0, 0.005)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: metric = {metric:.4f}")

        if es(metric, dummy_model, epoch):
            break

    print(f"\nStopped at epoch {epoch}")
    print(f"Best epoch: {es.best_epoch}")
    print(f"Best metric: {-es.best_score:.4f}")
    print("\nDone.")
```

---

### Script 4: Hyperparameter Search

Random search over model and training hyperparameters with abbreviated training.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Hyperparameter Search for MatBench Models.
Random search over LR, hidden_dim, layers, dropout, batch_size.
Runs abbreviated training (100 epochs) per configuration.
"""
import os, sys, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Search space ──────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "hidden_dim": [64, 128, 256],
    "n_conv_layers": [2, 3, 4, 5],
    "dropout": [0.0, 0.1, 0.2],
    "batch_size": [32, 64, 128],
    "weight_decay": [0, 1e-5, 1e-4],
}

# ── Settings ──────────────────────────────────────────────────────────────────
N_TRIALS = 20                  # Number of random configurations to try
SEARCH_EPOCHS = 100            # Abbreviated training per trial
TASK_NAME = "matbench_mp_e_form"
RESULTS_DIR = "/workspace/group/matbench/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def sample_config(search_space):
    """Sample a random configuration from the search space."""
    config = {}
    for key, values in search_space.items():
        config[key] = random.choice(values)
    return config


def build_model(config, atom_fea_dim=100, edge_fea_dim=50):
    """Build a CGCNN-like model from config. Replace with your model."""
    from torch_geometric.nn import CGConv, global_mean_pool

    class ConfigurableGNN(nn.Module):
        def __init__(self, config):
            super().__init__()
            hid = config["hidden_dim"]
            n_layers = config["n_conv_layers"]
            drop = config["dropout"]

            self.atom_embed = nn.Linear(atom_fea_dim, hid)
            self.convs = nn.ModuleList([
                CGConv(channels=hid, dim=edge_fea_dim, batch_norm=True)
                for _ in range(n_layers)
            ])
            self.fc1 = nn.Linear(hid, hid)
            self.fc2 = nn.Linear(hid, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(drop)

        def forward(self, data):
            x = self.relu(self.atom_embed(data.x))
            for conv in self.convs:
                x = self.relu(conv(x, data.edge_index, data.edge_attr))
            x = global_mean_pool(x, data.batch)
            x = self.dropout(self.relu(self.fc1(x)))
            return self.fc2(x).squeeze(-1)

    return ConfigurableGNN(config)


def train_trial(model, train_loader, val_loader, config, device,
                epochs=SEARCH_EPOCHS):
    """Run abbreviated training for one trial. Returns best val MAE."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.MSELoss()

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # Validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds.append(model(batch).cpu().numpy())
                trues.append(batch.y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mae = np.mean(np.abs(preds - trues))
        best_mae = min(best_mae, mae)

        scheduler.step()

        if epoch % 25 == 0:
            print(f"    Epoch {epoch}/{epochs}: MAE={mae:.5f} "
                  f"(best={best_mae:.5f})")

    return best_mae


# ── Main: run hyperparameter search ──────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"Hyperparameter Search: {N_TRIALS} trials, {SEARCH_EPOCHS} epochs each")
    print(f"Task: {TASK_NAME}")
    print(f"Search space: {json.dumps(SEARCH_SPACE, indent=2)}")
    print(f"{'='*60}\n")

    # NOTE: You need to provide train_graphs and val_graphs here.
    # Use the graph construction from structure-gnn/SKILL.md Script 1.
    # For this template, we show the search loop structure:

    print("IMPORTANT: This script needs pre-built graph datasets.")
    print("Use structure-gnn Script 1 to build graphs, then load them here.")
    print()
    print("Example integration:")
    print("  train_graphs = [torch.load(f) for f in sorted(glob('train/*.pt'))]")
    print("  val_graphs = [torch.load(f) for f in sorted(glob('val/*.pt'))]")
    print()

    # ── Search loop (template) ────────────────────────────────────────────
    # Uncomment and modify when you have real data:
    #
    # results = []
    # for trial in range(N_TRIALS):
    #     config = sample_config(SEARCH_SPACE)
    #     print(f"\nTrial {trial+1}/{N_TRIALS}: {config}")
    #
    #     train_loader = DataLoader(
    #         train_graphs, batch_size=config["batch_size"],
    #         shuffle=True, num_workers=4, pin_memory=True
    #     )
    #     val_loader = DataLoader(
    #         val_graphs, batch_size=config["batch_size"],
    #         shuffle=False, num_workers=4, pin_memory=True
    #     )
    #
    #     model = build_model(config)
    #     t0 = time.time()
    #     best_mae = train_trial(model, train_loader, val_loader, config, device)
    #     elapsed = time.time() - t0
    #
    #     result = {**config, "best_mae": best_mae, "time_sec": elapsed}
    #     results.append(result)
    #     print(f"  -> MAE={best_mae:.5f} ({elapsed:.0f}s)")
    #
    #     # Save incrementally
    #     results_path = os.path.join(RESULTS_DIR, f"hpsearch_{TASK_NAME}.json")
    #     with open(results_path, "w") as f:
    #         json.dump(results, f, indent=2)
    #
    # # Sort by MAE and print top 5
    # results.sort(key=lambda r: r["best_mae"])
    # print(f"\n{'='*60}")
    # print("Top 5 configurations:")
    # for i, r in enumerate(results[:5]):
    #     print(f"  {i+1}. MAE={r['best_mae']:.5f} | "
    #           f"lr={r['learning_rate']}, hid={r['hidden_dim']}, "
    #           f"layers={r['n_conv_layers']}, drop={r['dropout']}, "
    #           f"bs={r['batch_size']}")
    # print(f"\nResults saved: {results_path}")

    print("\nDone.")
```

---

### Script 5: TensorBoard Integration

Full TensorBoard logging setup for tracking MatBench training experiments.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
TensorBoard Integration for MatBench.
Log training loss, validation metrics, learning rate, and model graphs.
"""
import os, sys, time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── TensorBoard logger setup ─────────────────────────────────────────────────
def create_logger(model_name, task_name,
                  base_dir="/workspace/group/matbench/logs"):
    """Create a TensorBoard SummaryWriter with structured log directory.

    Log directory format: {base_dir}/{model}_{task}_{timestamp}/

    Args:
        model_name: e.g., "cgcnn", "schnet"
        task_name: e.g., "matbench_mp_e_form"
        base_dir: Base log directory

    Returns:
        SummaryWriter instance, log directory path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"{model_name}_{task_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Launch TensorBoard with:")
    print(f"  tensorboard --logdir {base_dir} --host 0.0.0.0 --port 6006")
    return writer, log_dir


# ── Training loop with TensorBoard logging ────────────────────────────────────
def train_with_tensorboard(model, train_loader, val_loader, device,
                           model_name="model", task_name="task",
                           epochs=300, lr=1e-3, fold=0):
    """Complete training loop with TensorBoard logging.

    Logs per epoch:
      - train/loss: training MSE loss
      - val/mae: validation mean absolute error
      - val/loss: validation MSE loss
      - lr: current learning rate
      - gpu/memory_allocated_gb: GPU memory usage
    """
    writer, log_dir = create_logger(
        f"{model_name}_fold{fold}", task_name
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.MSELoss()

    # Log hyperparameters
    writer.add_text("hyperparameters", (
        f"model={model_name}, task={task_name}, fold={fold}\n"
        f"epochs={epochs}, lr={lr}, optimizer=Adam\n"
        f"scheduler=CosineAnnealing, device={device}"
    ))

    best_mae = float("inf")
    ckpt_path = os.path.join(
        "/workspace/group/matbench/models",
        f"{model_name}_{task_name}_fold{fold}_best.pt"
    )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
        train_loss = total_loss / n

        # ── Validate ──
        model.eval()
        val_loss_total, val_n = 0.0, 0
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_loss_total += loss.item() * batch.num_graphs
                val_n += batch.num_graphs
                preds.append(pred.cpu().numpy())
                trues.append(batch.y.cpu().numpy())

        val_loss = val_loss_total / val_n
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_mae = np.mean(np.abs(preds - trues))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        # ── Log to TensorBoard ──
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mae", val_mae, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        if device.type == "cuda":
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            writer.add_scalar("gpu/memory_allocated_gb", gpu_mem, epoch)

        # ── Console logging ──
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss={train_loss:.5f} | "
                  f"Val MAE={val_mae:.5f} | "
                  f"LR={current_lr:.2e} | {elapsed:.1f}s")

        # ── Checkpoint ──
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)

    # Log best result
    writer.add_hparams(
        {"lr": lr, "epochs": epochs, "model": model_name},
        {"hparam/best_mae": best_mae},
    )
    writer.close()

    print(f"\nBest MAE: {best_mae:.5f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"TensorBoard logs: {log_dir}")

    return best_mae


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LOG_DIR = "/workspace/group/matbench/logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    print("TensorBoard Integration - Demo")
    print(f"Device: {device}")
    print()

    # Create a demo logger and write sample data
    writer, log_dir = create_logger("demo", "example_task")

    print("\nWriting sample training data to TensorBoard...")
    for epoch in range(1, 101):
        train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
        val_mae = 0.5 / (epoch + 1) + np.random.normal(0, 0.005) + 0.02
        lr = 1e-3 * (1 + np.cos(np.pi * epoch / 100)) / 2

        writer.add_scalar("train/loss", max(0, train_loss), epoch)
        writer.add_scalar("val/mae", max(0, val_mae), epoch)
        writer.add_scalar("lr", lr, epoch)

        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss={train_loss:.4f}, mae={val_mae:.4f}")

    writer.close()

    print(f"\nDemo data written to: {log_dir}")
    print(f"\nTo view TensorBoard, run:")
    print(f"  tensorboard --logdir {LOG_DIR} --host 0.0.0.0 --port 6006")
    print(f"\nThen open http://localhost:6006 in your browser.")
    print("\nDone.")
```

---

### Script 6: Mixed Precision (AMP) on A100

Automatic mixed precision (AMP) training for up to 2x speedup on A100 GPU.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Mixed Precision (AMP) Training on A100 for MatBench.
Uses torch.cuda.amp for FP16 training with ~2x speedup.
A100 also supports TF32 for additional performance gains.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── A100 Optimizations ───────────────────────────────────────────────────────
def setup_a100_optimizations():
    """Enable A100-specific performance optimizations.

    A100 features:
      - TF32: Enabled by default for matmul and cuDNN. ~3x faster than FP32
        with minimal accuracy loss. No code changes needed.
      - FP16 AMP: Explicit opt-in via autocast + GradScaler. ~2x speedup.
      - Combined: TF32 for FP32 ops + FP16 for eligible ops = best speed.
    """
    if not torch.cuda.is_available():
        print("No CUDA device. Running on CPU (no AMP).")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_mem:.1f} GB")

    # TF32 is enabled by default on A100, verify:
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cuDNN:  {torch.backends.cudnn.allow_tf32}")

    # Ensure TF32 is enabled (should be default)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark mode: auto-tune conv algorithms
    torch.backends.cudnn.benchmark = True

    print("A100 optimizations enabled: TF32 + cuDNN benchmark")


# ── AMP Training Loop ────────────────────────────────────────────────────────
def train_with_amp(model, train_loader, val_loader, device,
                   epochs=300, lr=1e-3, grad_clip=10.0,
                   save_path="/workspace/group/matbench/models/amp_best.pt"):
    """Training loop with Automatic Mixed Precision (AMP).

    Uses torch.cuda.amp.autocast for FP16 forward/backward pass
    and GradScaler for stable gradient updates.

    Speedup: ~2x on A100 compared to pure FP32.
    Accuracy: Negligible difference for MatBench tasks.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.MSELoss()

    # AMP components
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_mae = float("inf")

    print(f"AMP enabled: {use_amp}")
    print(f"Training for {epochs} epochs, LR={lr}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train with AMP ──
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # autocast: eligible ops run in FP16, others stay FP32
            with autocast(device_type="cuda", enabled=use_amp):
                pred = model(batch)
                loss = criterion(pred, batch.y)

            # GradScaler: scale loss to prevent FP16 underflow
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )

            # Step optimizer with scaled gradients
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs

        train_loss = total_loss / n

        # ── Validate with AMP (no grad scaling needed) ──
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                with autocast(device_type="cuda", enabled=use_amp):
                    pred = model(batch)
                preds.append(pred.float().cpu().numpy())
                trues.append(batch.y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_mae = np.mean(np.abs(preds - trues))

        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            mem_gb = torch.cuda.memory_allocated() / 1e9 if use_amp else 0
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss={train_loss:.5f} | MAE={val_mae:.5f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                  f"Mem={mem_gb:.1f}GB | {elapsed:.1f}s")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_path)

    print(f"\nBest MAE: {best_mae:.5f}")
    print(f"Checkpoint: {save_path}")
    return best_mae


# ── Batch size recommendations for 80GB A100 VRAM ────────────────────────────
BATCH_SIZE_GUIDE = """
Batch Size Recommendations for A100-SXM4-80GB:

| Model      | FP32 batch | AMP batch | Notes                          |
|------------|------------|-----------|--------------------------------|
| CGCNN      | 256        | 512       | Lightweight, memory-efficient  |
| SchNet     | 128        | 256       | Medium memory, filter convs    |
| DimeNet++  | 32         | 64-128    | Heavy, triplet interactions    |
| RF/sklearn | N/A        | N/A       | CPU-based, no GPU needed       |
| MLP        | 1024       | 2048      | Dense layers, very efficient   |

Notes:
- These are estimates for typical MatBench crystal sizes (5-100 atoms)
- Larger crystals (>100 atoms) need smaller batch sizes
- Monitor with torch.cuda.memory_allocated() during training
- If OOM: halve batch_size, or use gradient accumulation
"""


# ── Gradient accumulation (for effective large batch on limited memory) ───────
def train_epoch_with_accumulation(model, train_loader, optimizer, criterion,
                                  scaler, device, accumulation_steps=4):
    """Train one epoch with gradient accumulation.

    Simulates a larger batch size without extra memory.
    Effective batch size = batch_size * accumulation_steps

    Example: batch_size=32, accumulation_steps=4 -> effective batch=128
    """
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch = batch.to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss = loss / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * batch.num_graphs
        n += batch.num_graphs

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / n


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Mixed Precision (AMP) Training on A100 - Demo")
    print()

    setup_a100_optimizations()
    print()
    print(BATCH_SIZE_GUIDE)

    # Benchmark AMP vs FP32 with a dummy model
    if device.type == "cuda":
        print("Benchmarking AMP vs FP32...\n")

        # Dummy model
        model = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        ).to(device)

        x = torch.randn(1024, 256, device=device)
        y = torch.randn(1024, 1, device=device)
        criterion = nn.MSELoss()

        # FP32
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        fp32_time = time.time() - t0

        # AMP
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        amp_time = time.time() - t0

        print(f"FP32: {fp32_time:.3f}s for 100 iterations")
        print(f"AMP:  {amp_time:.3f}s for 100 iterations")
        print(f"Speedup: {fp32_time / amp_time:.2f}x")
    else:
        print("No GPU available. AMP requires CUDA.")

    print("\nDone.")
```

---

## Key Parameters

| Parameter | Description | Typical Range | Default |
|---|---|---|---|
| `learning_rate` | Initial learning rate | 1e-4 to 5e-3 | 1e-3 |
| `weight_decay` | L2 regularization strength | 0 to 1e-4 | 1e-5 |
| `epochs` | Maximum training epochs | 100 to 500 | 300 |
| `patience` | Early stopping patience (epochs) | 20 to 100 | 50 |
| `batch_size` | Samples per training batch | 32 to 512 | 128 |
| `scheduler` | LR schedule type | cosine, plateau, onecycle, warmup | cosine |
| `grad_clip_norm` | Maximum gradient norm | 1.0 to 100.0 | 10.0 |
| `amp_enabled` | Use mixed precision (FP16) | True / False | True on GPU |
| `num_workers` | DataLoader worker processes | 0 to 8 | 4 |
| `accumulation_steps` | Gradient accumulation steps | 1 to 8 | 1 |

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| **NaN loss** | LR too high, unstable gradients | Reduce LR by 10x. Enable gradient clipping. Check input data for NaN/Inf values. With AMP, ensure GradScaler is used correctly. |
| **Overfitting** | Model too large for dataset, no regularization | Add dropout (0.1-0.2). Increase weight_decay (1e-4). Use fewer layers. Enable early stopping. |
| **CUDA out of memory** | Batch too large for 80GB VRAM | Halve batch_size. Use gradient accumulation to maintain effective batch size. Call `torch.cuda.empty_cache()` between folds. |
| **TensorBoard shows no data** | Writer not flushed, wrong log directory | Call `writer.flush()` periodically. Verify `--logdir` matches the log directory. Ensure `writer.close()` at the end. |
| **Slow training** | No AMP, no pin_memory, wrong num_workers | Enable AMP (Script 6). Set `pin_memory=True` in DataLoader. Tune `num_workers` (4 is a good start). Enable cuDNN benchmark mode. |
| **Early stopping too early** | Patience too low, noisy validation | Increase patience (50-100). Use larger validation set. Smooth val metric with running average. |
| **Learning rate too high/low** | Wrong schedule or initial LR | Use LR finder: train for a few epochs at increasing LR, pick LR where loss decreases fastest. Or use OneCycleLR which auto-adjusts. |
| **Gradient accumulation mismatch** | Loss not normalized by accumulation steps | Divide loss by `accumulation_steps` before `.backward()`. See Script 6 `train_epoch_with_accumulation()`. |
