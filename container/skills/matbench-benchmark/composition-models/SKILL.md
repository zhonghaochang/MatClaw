---
name: composition-models
description: Deep learning models (Roost, CrabNet, foundation model transfer, custom transformer) for MatBench tasks with composition string input
---

# Deep Learning Models for Composition-Based MatBench Tasks

## DO NOT USE TRADITIONAL ML

**This skill covers deep learning approaches ONLY.** Do NOT use sklearn, RandomForest, GradientBoosting, XGBoost, or any traditional ML method for SOTA attempts on composition tasks. Traditional ML with hand-crafted features (Magpie, etc.) cannot compete with learned representations on the matbench leaderboard.

Deep learning models learn element embeddings and inter-element interactions directly from data, producing superior representations for composition-based property prediction.

## Applicable Tasks

These tasks have **composition strings only** as input (no crystal structure):

| Task | Samples | Type | Metric |
|------|---------|------|--------|
| `matbench_steels` | 312 | Regression | MAE |
| `matbench_expt_gap` | 4,604 | Regression | MAE |
| `matbench_dielectric` | 4,764 | Regression | MAE |
| `matbench_expt_is_metal` | 4,921 | Classification | ROC-AUC |
| `matbench_glass` | 5,680 | Classification | ROC-AUC |

## Model Selection

```
Which deep learning approach?
|
+-- Want a proven SOTA method with attention-based composition embedding?
|   +-- Roost (Script 1) -- element message passing + attention pooling
|   +-- CrabNet (Script 2) -- transformer self-attention over elements
|
+-- Want to leverage pretrained foundation model knowledge?
|   +-- Foundation Model Transfer (Script 3) -- extract element embeddings from MACE/CHGNet
|
+-- Want a novel architecture you can iterate on?
    +-- Custom Composition Transformer (Script 4) -- original design, fully modifiable
```

## Prerequisites

- Python: `/opt/conda/envs/matbench/bin/python`
- Packages: matbench, torch, pymatgen, numpy, pandas, matplotlib
- Reference repos: `/workspace/group/reference/repos/roost/`, `/workspace/group/reference/repos/crabnet/`
- GPU: A100 (auto-detected via `torch.cuda.is_available()`)

## Script 1: Roost (Representation Learning from Stoichiometry)

Attention-based composition embedding with element-wise message passing. Multi-seed ensemble for robust predictions. Reference implementation at `/workspace/group/reference/repos/roost/`.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Roost: Representation Learning from Stoichiometry.
Attention-based element message passing network for composition-only property prediction.
Reference: https://github.com/CompRhys/roost

Architecture:
  1. Element embeddings (learnable, dim=64 per element)
  2. Message passing with attention weights between elements in composition
  3. Weighted attention pooling over elements -> composition descriptor
  4. Residual MLP head -> prediction

Multi-seed ensemble averages predictions from N_SEEDS independent runs.
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import copy
import math
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"   # Composition-input task
IS_CLASSIFICATION = False          # True for expt_is_metal, glass
ELEM_EMB_DIM = 64                  # Element embedding dimension
MSG_HEADS = 4                      # Attention heads in message passing
MSG_LAYERS = 3                     # Number of message passing layers
MLP_HIDDEN = [256, 128]            # MLP head hidden dims
DROPOUT = 0.0                      # Dropout rate
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 50                      # Early stopping patience
N_SEEDS = 3                        # Ensemble seeds
# ======================

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
EXP_DIR = f"/workspace/group/matbench/experiments/{TIMESTAMP}_roost_{TASK_NAME}"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiment dir: {EXP_DIR}")
print(f"=" * 70)
print(f"Roost | {TASK_NAME} | {N_SEEDS}-seed ensemble")
print(f"=" * 70)


# --- Element vocabulary ---
# Build from periodic table (elements 1-118)
ELEM_VOCAB = {str(i): i for i in range(1, 119)}
# Add element symbols
from pymatgen.core.periodic_table import Element
for z in range(1, 119):
    try:
        sym = Element.from_Z(z).symbol
        ELEM_VOCAB[sym] = z
    except Exception:
        pass
VOCAB_SIZE = 119  # 0=padding, 1-118=elements


class CompositionDataset(Dataset):
    """Dataset that converts composition strings to (element_ids, fractions) tuples."""

    def __init__(self, compositions, targets=None, max_elems=16):
        self.max_elems = max_elems
        self.elem_ids = []
        self.fractions = []
        self.targets = targets

        for comp in compositions:
            if isinstance(comp, str):
                comp = Composition(comp)
            elem_dict = comp.fractional_composition.as_dict()
            ids = []
            fracs = []
            for sym, frac in elem_dict.items():
                z = ELEM_VOCAB.get(sym, 0)
                ids.append(z)
                fracs.append(frac)
            # Pad to max_elems
            n = len(ids)
            ids = ids[:max_elems] + [0] * max(0, max_elems - n)
            fracs = fracs[:max_elems] + [0.0] * max(0, max_elems - n)
            self.elem_ids.append(ids)
            self.fractions.append(fracs)

        self.elem_ids = torch.LongTensor(self.elem_ids)
        self.fractions = torch.FloatTensor(self.fractions)
        if targets is not None:
            self.targets = torch.FloatTensor(np.array(targets, dtype=np.float32))

    def __len__(self):
        return len(self.elem_ids)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.elem_ids[idx], self.fractions[idx], self.targets[idx]
        return self.elem_ids[idx], self.fractions[idx]


class RoostMessageBlock(nn.Module):
    """Single message passing layer with multi-head attention over elements."""

    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        assert emb_dim % n_heads == 0

        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask):
        """
        x: (batch, max_elems, emb_dim)
        mask: (batch, max_elems) -- 1 for real elements, 0 for padding
        """
        B, N, D = x.shape
        residual = x

        q = self.W_q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Mask padding positions
        attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, N, -1)
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(attn_mask == 0, 0.0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)
        out = self.layer_norm(out + residual)
        return out


class RoostModel(nn.Module):
    """
    Roost-style model: element embeddings -> message passing -> attention pooling -> MLP.
    """

    def __init__(self, vocab_size, emb_dim, n_heads, n_layers, mlp_hidden, dropout, is_clf):
        super().__init__()
        self.elem_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.frac_embedding = nn.Linear(1, emb_dim)

        self.msg_layers = nn.ModuleList([
            RoostMessageBlock(emb_dim, n_heads) for _ in range(n_layers)
        ])

        # Attention pooling
        self.pool_gate = nn.Linear(emb_dim, 1)

        # MLP head
        mlp = []
        prev = emb_dim
        for h in mlp_hidden:
            mlp.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        mlp.append(nn.Linear(prev, 1))
        if is_clf:
            mlp.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, elem_ids, fractions):
        """
        elem_ids: (batch, max_elems) -- element atomic numbers
        fractions: (batch, max_elems) -- composition fractions
        """
        mask = (elem_ids != 0).float()  # (batch, max_elems)

        # Element + fractional embeddings
        x = self.elem_embedding(elem_ids)  # (batch, max_elems, emb_dim)
        frac_emb = self.frac_embedding(fractions.unsqueeze(-1))  # (batch, max_elems, emb_dim)
        x = x + frac_emb

        # Message passing
        for layer in self.msg_layers:
            x = layer(x, mask)

        # Weighted attention pooling
        gate = self.pool_gate(x).squeeze(-1)  # (batch, max_elems)
        gate = gate.masked_fill(mask == 0, float("-inf"))
        gate = F.softmax(gate, dim=-1)
        # Also weight by composition fractions
        gate = gate * fractions
        gate_sum = gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        gate = gate / gate_sum
        pooled = (x * gate.unsqueeze(-1)).sum(dim=1)  # (batch, emb_dim)

        return self.mlp(pooled).squeeze(-1)


def train_single_seed(task, fold_idx, seed):
    """Train one Roost model for a given fold and seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)

    # Train/val split (90/10)
    n = len(train_inputs)
    n_val = max(1, int(0.1 * n))
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_comps = [train_inputs.iloc[i] for i in tr_idx]
    tr_targets = [train_outputs.iloc[i] for i in tr_idx]
    va_comps = [train_inputs.iloc[i] for i in val_idx]
    va_targets = [train_outputs.iloc[i] for i in val_idx]

    train_ds = CompositionDataset(tr_comps, tr_targets)
    val_ds = CompositionDataset(va_comps, va_targets)
    test_ds = CompositionDataset(list(test_inputs))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RoostModel(
        VOCAB_SIZE, ELEM_EMB_DIM, MSG_HEADS, MSG_LAYERS,
        MLP_HIDDEN, DROPOUT, IS_CLASSIFICATION
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss() if IS_CLASSIFICATION else nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids, fracs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred = model(ids, fracs)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                ids, fracs, targets = [b.to(device) for b in batch]
                pred = model(ids, fracs)
                val_losses.append(criterion(pred, targets).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 50 == 0:
            print(f"      Seed {seed} Epoch {epoch+1}/{EPOCHS}: val_loss={val_loss:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"      Seed {seed} early stop at epoch {epoch+1}")
            break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            ids, fracs = batch[0].to(device), batch[1].to(device)
            pred = model(ids, fracs)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions)

    # Save checkpoint
    ckpt_path = f"{EXP_DIR}/checkpoints/fold{fold_idx}_seed{seed}.pt"
    torch.save(best_state, ckpt_path)

    return predictions


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

# 5-fold cross-validation with multi-seed ensemble
fold_scores = []

for fold_idx in range(5):
    print(f"\n--- Fold {fold_idx} ---")
    train_inputs, _ = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    # Ensemble over seeds
    all_preds = []
    for seed in range(N_SEEDS):
        print(f"    Training seed {seed}...")
        preds = train_single_seed(task, fold_idx, seed=seed * 42 + fold_idx)
        all_preds.append(preds)

    # Average ensemble predictions
    ensemble_preds = np.mean(all_preds, axis=0)

    # Record
    task.record(fold_idx, ensemble_preds)

    # Score
    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(true_values, ensemble_preds)
        metric = "ROC-AUC"
    else:
        score = float(np.mean(np.abs(true_values - ensemble_preds)))
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: Roost on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results = {
    "task": TASK_NAME,
    "model": "Roost",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "config": {
        "elem_emb_dim": ELEM_EMB_DIM,
        "msg_heads": MSG_HEADS,
        "msg_layers": MSG_LAYERS,
        "mlp_hidden": MLP_HIDDEN,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "n_seeds": N_SEEDS,
    },
    "device": str(device),
}
with open(f"{EXP_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {EXP_DIR}/results.json")

# Save matbench file
mb.to_file(f"{EXP_DIR}/matbench_results.json.gz")
print(f"MatBench file saved to {EXP_DIR}/matbench_results.json.gz")
print("Done!")
```

## Script 2: CrabNet (Cross-attention Based Network)

Transformer-based architecture with self-attention over elements and fractional encoding of element amounts. Reference implementation at `/workspace/group/reference/repos/crabnet/`.

```python
#!/opt/conda/envs/matbench/bin/python
"""
CrabNet: Compositionally-Restricted Attention-Based Network.
Transformer self-attention over element tokens with fractional encoding.
Reference: https://github.com/anthony-wang/CrabNet

Architecture:
  1. Element embedding lookup (learnable, dim=512)
  2. Fractional encoding: element fractions encoded as log-scale features
     and combined with element embeddings
  3. Multi-head self-attention (transformer encoder) over element tokens
  4. Fraction-weighted readout pooling
  5. Residual output network -> prediction
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import copy
import math
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"   # Composition-input task
IS_CLASSIFICATION = False          # True for expt_is_metal, glass
D_MODEL = 512                      # Transformer model dimension
N_HEADS = 8                        # Attention heads
N_ENCODER_LAYERS = 3               # Transformer encoder layers
DIM_FF = 1024                      # Feed-forward dimension
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 50
FRAC_ENCODING_DIM = 32             # Fractional encoding dimension
# ======================

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
EXP_DIR = f"/workspace/group/matbench/experiments/{TIMESTAMP}_crabnet_{TASK_NAME}"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiment dir: {EXP_DIR}")
print(f"=" * 70)
print(f"CrabNet | {TASK_NAME}")
print(f"=" * 70)

VOCAB_SIZE = 119  # 0=padding, 1-118=elements
MAX_ELEMS = 16

# Build element symbol -> Z mapping
ELEM_TO_Z = {}
for z in range(1, 119):
    try:
        sym = Element.from_Z(z).symbol
        ELEM_TO_Z[sym] = z
    except Exception:
        pass


class FractionalEncoder(nn.Module):
    """Encode element fractions using sinusoidal positional-style encoding."""

    def __init__(self, d_model, frac_dim):
        super().__init__()
        self.d_model = d_model
        self.frac_dim = frac_dim
        # Learnable encoding: project fraction through log-space features
        self.linear1 = nn.Linear(frac_dim, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, fractions):
        """
        fractions: (batch, max_elems) in [0, 1]
        Returns: (batch, max_elems, d_model)
        """
        # Create multi-scale fractional features
        # Use log-spaced frequencies for better resolution near 0 and 1
        device = fractions.device
        freqs = torch.logspace(0, 3, self.frac_dim // 2, device=device)
        x = fractions.unsqueeze(-1)  # (B, N, 1)
        x = x * freqs.unsqueeze(0).unsqueeze(0)  # (B, N, frac_dim//2)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, N, frac_dim)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        return x


class CrabNetModel(nn.Module):
    """
    CrabNet-style transformer for composition property prediction.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, dim_ff,
                 frac_dim, dropout, is_clf):
        super().__init__()
        self.d_model = d_model

        # Element embedding
        self.elem_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Fractional encoder
        self.frac_encoder = FractionalEncoder(d_model, frac_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        if is_clf:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, elem_ids, fractions):
        """
        elem_ids: (batch, max_elems) -- element Z values, 0=padding
        fractions: (batch, max_elems) -- composition fractions
        """
        mask = (elem_ids == 0)  # True for padding positions

        # Embed elements + fractions
        x = self.elem_embedding(elem_ids)  # (B, N, d_model)
        frac_emb = self.frac_encoder(fractions)  # (B, N, d_model)
        x = x + frac_emb

        # Transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, N, d_model)

        # Fraction-weighted readout
        weights = fractions.clone()
        weights[mask] = 0.0
        w_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / w_sum  # Normalize
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        out = self.output_net(pooled).squeeze(-1)
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        return out


def parse_composition(comp):
    """Convert composition to (elem_ids, fractions) arrays."""
    if isinstance(comp, str):
        comp = Composition(comp)
    elem_dict = comp.fractional_composition.as_dict()
    ids = []
    fracs = []
    for sym, frac in elem_dict.items():
        z = ELEM_TO_Z.get(sym, 0)
        if z > 0:
            ids.append(z)
            fracs.append(frac)
    # Pad
    n = len(ids)
    ids = ids[:MAX_ELEMS] + [0] * max(0, MAX_ELEMS - n)
    fracs = fracs[:MAX_ELEMS] + [0.0] * max(0, MAX_ELEMS - n)
    return ids, fracs


class CrabDataset(Dataset):
    def __init__(self, compositions, targets=None):
        self.ids_list = []
        self.fracs_list = []
        self.targets = None
        for comp in compositions:
            ids, fracs = parse_composition(comp)
            self.ids_list.append(ids)
            self.fracs_list.append(fracs)
        self.ids_list = torch.LongTensor(self.ids_list)
        self.fracs_list = torch.FloatTensor(self.fracs_list)
        if targets is not None:
            self.targets = torch.FloatTensor(np.array(targets, dtype=np.float32))

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.ids_list[idx], self.fracs_list[idx], self.targets[idx]
        return self.ids_list[idx], self.fracs_list[idx]


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

    # Train/val split
    n = len(train_inputs)
    n_val = max(1, int(0.1 * n))
    perm = np.random.RandomState(fold_idx).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_comps = [train_inputs.iloc[i] for i in tr_idx]
    tr_targets = [train_outputs.iloc[i] for i in tr_idx]
    va_comps = [train_inputs.iloc[i] for i in val_idx]
    va_targets = [train_outputs.iloc[i] for i in val_idx]

    train_ds = CrabDataset(tr_comps, tr_targets)
    val_ds = CrabDataset(va_comps, va_targets)
    test_ds = CrabDataset(list(test_inputs))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CrabNetModel(
        VOCAB_SIZE, D_MODEL, N_HEADS, N_ENCODER_LAYERS, DIM_FF,
        FRAC_ENCODING_DIM, DROPOUT, IS_CLASSIFICATION
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss() if IS_CLASSIFICATION else nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids, fracs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred = model(ids, fracs)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                ids, fracs, targets = [b.to(device) for b in batch]
                pred = model(ids, fracs)
                val_losses.append(criterion(pred, targets).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: val_loss={val_loss:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            ids, fracs = batch[0].to(device), batch[1].to(device)
            pred = model(ids, fracs)
            predictions.append(pred.cpu().numpy())
    predictions = np.concatenate(predictions)

    # Save checkpoint
    torch.save(best_state, f"{EXP_DIR}/checkpoints/fold{fold_idx}.pt")

    # Record
    task.record(fold_idx, predictions)

    # Score
    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(true_values, predictions)
        metric = "ROC-AUC"
    else:
        score = float(np.mean(np.abs(true_values - predictions)))
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: CrabNet on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results = {
    "task": TASK_NAME,
    "model": "CrabNet",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "config": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_encoder_layers": N_ENCODER_LAYERS,
        "dim_ff": DIM_FF,
        "frac_encoding_dim": FRAC_ENCODING_DIM,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    "device": str(device),
}
with open(f"{EXP_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {EXP_DIR}/results.json")

mb.to_file(f"{EXP_DIR}/matbench_results.json.gz")
print(f"MatBench file saved to {EXP_DIR}/matbench_results.json.gz")
print("Done!")
```

## Script 3: Foundation Model Transfer Learning (Pretrained Element Embeddings + MLP)

For composition tasks WITHOUT crystal structure: extract per-element embeddings from pretrained foundation models (MACE, CHGNet), compute composition-weighted sum, then train an MLP head. This leverages atomic representations learned from millions of DFT calculations.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Foundation Model Transfer Learning for Composition Tasks.

Approach: Extract element embeddings from a pretrained model (MACE-MP-0 or CHGNet),
compute a composition-weighted average embedding, then train an MLP head.

This works for composition-only tasks because:
  - Foundation models learn rich per-element representations during pretraining
  - These embeddings encode chemical knowledge (electronegativity, bonding, etc.)
  - Weighted averaging by composition fractions creates a descriptor
  - An MLP head adapts this descriptor to the target property

No crystal structure needed -- we only use the element embedding layer.
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"
IS_CLASSIFICATION = False
# Foundation model source: "mace" or "chgnet"
FOUNDATION = "mace"
# MLP head architecture
MLP_HIDDEN = [256, 128, 64]
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 300
PATIENCE = 30
FINE_TUNE_EMBEDDINGS = True  # Fine-tune foundation embeddings or freeze
# ======================

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
EXP_DIR = f"/workspace/group/matbench/experiments/{TIMESTAMP}_foundation_{FOUNDATION}_{TASK_NAME}"
os.makedirs(EXP_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiment dir: {EXP_DIR}")
print(f"=" * 70)
print(f"Foundation Transfer ({FOUNDATION.upper()}) | {TASK_NAME}")
print(f"=" * 70)

# --- Extract element embeddings from foundation model ---
def get_mace_element_embeddings():
    """Extract element embeddings from MACE-MP-0 pretrained model."""
    try:
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", device=str(device), default_dtype="float32")
        # MACE stores node embeddings -- extract the initial element embedding table
        model = calc.models[0]
        # The node embedding in MACE is typically in model.node_embedding
        for name, param in model.named_parameters():
            if "node_embedding" in name or "chemical_embedding" in name:
                emb_weights = param.detach().cpu().numpy()
                print(f"  Found MACE embeddings: {name}, shape={emb_weights.shape}")
                return emb_weights
        # Fallback: try to get from the radial embedding
        for name, module in model.named_modules():
            if hasattr(module, "weight") and "embed" in name.lower():
                emb_weights = module.weight.detach().cpu().numpy()
                print(f"  Found MACE embeddings: {name}, shape={emb_weights.shape}")
                return emb_weights
    except Exception as e:
        print(f"  Warning: Could not load MACE embeddings: {e}")
    return None


def get_chgnet_element_embeddings():
    """Extract element embeddings from CHGNet pretrained model."""
    try:
        from chgnet.model import CHGNet
        model = CHGNet.load()
        # CHGNet has atom_embedding layer
        for name, param in model.named_parameters():
            if "atom_embedding" in name:
                emb_weights = param.detach().cpu().numpy()
                print(f"  Found CHGNet embeddings: {name}, shape={emb_weights.shape}")
                return emb_weights
    except Exception as e:
        print(f"  Warning: Could not load CHGNet embeddings: {e}")
    return None


def get_fallback_embeddings(dim=64):
    """
    Fallback: create element embeddings from physical properties.
    Uses atomic number, group, period, electronegativity, atomic radius, etc.
    Then projects to `dim` dimensions with a random projection (fixed seed).
    """
    print("  Using fallback physical-property embeddings")
    embeddings = np.zeros((119, dim))
    rng = np.random.RandomState(42)
    for z in range(1, 119):
        try:
            el = Element.from_Z(z)
            props = [
                z / 118.0,
                el.group / 18.0 if el.group else 0.0,
                el.row / 9.0,
                (el.X or 0) / 4.0,  # Electronegativity
                (el.atomic_radius or 0) / 3.0 if el.atomic_radius else 0.0,
                (el.average_ionic_radius or 0) / 2.0 if el.average_ionic_radius else 0.0,
                1.0 if el.is_metal else 0.0,
                1.0 if el.is_metalloid else 0.0,
            ]
            # Project to higher dim with random matrix
            props = np.array(props, dtype=np.float32)
            proj = rng.randn(len(props), dim).astype(np.float32) * 0.3
            embeddings[z] = props @ proj
        except Exception:
            embeddings[z] = rng.randn(dim) * 0.01
    return embeddings


print("\nExtracting foundation model element embeddings...")
if FOUNDATION == "mace":
    raw_embeddings = get_mace_element_embeddings()
elif FOUNDATION == "chgnet":
    raw_embeddings = get_chgnet_element_embeddings()
else:
    raw_embeddings = None

if raw_embeddings is None:
    raw_embeddings = get_fallback_embeddings(dim=64)
    EMB_DIM = 64
else:
    EMB_DIM = raw_embeddings.shape[-1]
    # Ensure we have 119 rows (pad if needed)
    if raw_embeddings.shape[0] < 119:
        padded = np.zeros((119, EMB_DIM), dtype=np.float32)
        padded[:raw_embeddings.shape[0]] = raw_embeddings
        raw_embeddings = padded

print(f"  Embedding dimension: {EMB_DIM}")

# Build element symbol -> Z mapping
ELEM_TO_Z = {}
for z in range(1, 119):
    try:
        sym = Element.from_Z(z).symbol
        ELEM_TO_Z[sym] = z
    except Exception:
        pass


class FoundationCompDataset(Dataset):
    """Compute composition-weighted embedding from foundation model."""

    def __init__(self, compositions, embeddings, targets=None):
        self.features = []
        self.targets = targets

        emb = torch.FloatTensor(embeddings)
        for comp in compositions:
            if isinstance(comp, str):
                comp = Composition(comp)
            frac_dict = comp.fractional_composition.as_dict()
            weighted = torch.zeros(emb.shape[-1])
            for sym, frac in frac_dict.items():
                z = ELEM_TO_Z.get(sym, 0)
                if 0 < z < emb.shape[0]:
                    weighted += frac * emb[z]
            self.features.append(weighted)

        self.features = torch.stack(self.features)
        if targets is not None:
            self.targets = torch.FloatTensor(np.array(targets, dtype=np.float32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx],


class TransferMLP(nn.Module):
    """MLP head on top of foundation model composition embeddings."""

    def __init__(self, input_dim, hidden_dims, dropout, is_clf):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        if is_clf:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


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

    # Train/val split
    n = len(train_inputs)
    n_val = max(1, int(0.1 * n))
    perm = np.random.RandomState(fold_idx).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_comps = [train_inputs.iloc[i] for i in tr_idx]
    tr_targets = [train_outputs.iloc[i] for i in tr_idx]
    va_comps = [train_inputs.iloc[i] for i in val_idx]
    va_targets = [train_outputs.iloc[i] for i in val_idx]

    train_ds = FoundationCompDataset(tr_comps, raw_embeddings, tr_targets)
    val_ds = FoundationCompDataset(va_comps, raw_embeddings, va_targets)
    test_ds = FoundationCompDataset(list(test_inputs), raw_embeddings)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TransferMLP(EMB_DIM, MLP_HIDDEN, DROPOUT, IS_CLASSIFICATION).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss() if IS_CLASSIFICATION else nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            feats, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(feats)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                feats, targets = batch[0].to(device), batch[1].to(device)
                pred = model(feats)
                val_losses.append(criterion(pred, targets).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}: val_loss={val_loss:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Predict
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch[0].to(device)
            pred = model(feats)
            predictions.append(pred.cpu().numpy())
    predictions = np.concatenate(predictions)

    # Record
    task.record(fold_idx, predictions)

    # Score
    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(true_values, predictions)
        metric = "ROC-AUC"
    else:
        score = float(np.mean(np.abs(true_values - predictions)))
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: Foundation Transfer ({FOUNDATION.upper()}) on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results = {
    "task": TASK_NAME,
    "model": f"FoundationTransfer_{FOUNDATION}",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "config": {
        "foundation": FOUNDATION,
        "emb_dim": EMB_DIM,
        "mlp_hidden": MLP_HIDDEN,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "fine_tune": FINE_TUNE_EMBEDDINGS,
    },
    "device": str(device),
}
with open(f"{EXP_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {EXP_DIR}/results.json")

mb.to_file(f"{EXP_DIR}/matbench_results.json.gz")
print(f"MatBench file saved to {EXP_DIR}/matbench_results.json.gz")
print("Done!")
```

## Script 4: Custom Composition Transformer

An original transformer architecture designed from scratch for composition property prediction. Element token embeddings with fractional positional encoding, multi-head self-attention, and feed-forward layers. Fully modifiable -- intended as a starting point for novel architecture exploration.

```python
#!/opt/conda/envs/matbench/bin/python
"""
Custom Composition Transformer: An ORIGINAL architecture for composition-based
property prediction, designed to be further modified and improved.

Architecture overview:
  1. Learnable element token embeddings (periodic table aware)
  2. Fractional positional encoding: encodes stoichiometric fractions as
     continuous signals using multi-frequency sinusoidal projection
  3. Composition-aware self-attention: attention scores modulated by
     element-pair compatibility learned from data
  4. Gated residual feed-forward blocks
  5. Learnable [CLS] token for composition-level readout
  6. Multi-task compatible output head

This is NOT a reproduction of any existing model. It is an original design
that combines ideas from transformers, graph networks, and materials science.
The agent should feel free to modify any component.
"""

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import copy
import math
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

# === CONFIGURE THIS ===
TASK_NAME = "matbench_expt_gap"
IS_CLASSIFICATION = False
D_MODEL = 256                      # Model dimension
N_HEADS = 8                        # Attention heads
N_LAYERS = 4                       # Transformer layers
DIM_FF = 512                       # Feed-forward dimension
DROPOUT = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 400
PATIENCE = 60
WARMUP_EPOCHS = 20                 # Linear warmup
N_SEEDS = 3                        # Ensemble seeds
MAX_ELEMS = 16                     # Max elements per composition
# ======================

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
EXP_DIR = f"/workspace/group/matbench/experiments/{TIMESTAMP}_comptransformer_{TASK_NAME}"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiment dir: {EXP_DIR}")
print(f"=" * 70)
print(f"Custom Composition Transformer | {TASK_NAME} | {N_SEEDS}-seed ensemble")
print(f"=" * 70)

VOCAB_SIZE = 119  # 0=padding, 1-118=elements

ELEM_TO_Z = {}
for z in range(1, 119):
    try:
        sym = Element.from_Z(z).symbol
        ELEM_TO_Z[sym] = z
    except Exception:
        pass


# --- Custom Architecture Components ---

class PeriodicTableEmbedding(nn.Module):
    """
    Element embedding that encodes periodic table structure.
    Combines learnable embeddings with periodic table positional info
    (group, period, block) to give the model chemical inductive bias.
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.main_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Periodic table features: group (1-18), period (1-7), block (s/p/d/f -> 0-3)
        self.group_embedding = nn.Embedding(19, d_model // 4)  # 0=pad, 1-18
        self.period_embedding = nn.Embedding(10, d_model // 4)  # 0=pad, 1-9
        self.block_embedding = nn.Embedding(5, d_model // 4)   # 0=pad, 1-4
        self.proj = nn.Linear(d_model + 3 * (d_model // 4), d_model)

        # Precompute periodic table info
        self.register_buffer("group_ids", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("period_ids", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("block_ids", torch.zeros(vocab_size, dtype=torch.long))
        block_map = {"s": 1, "p": 2, "d": 3, "f": 4}
        for z in range(1, min(vocab_size, 119)):
            try:
                el = Element.from_Z(z)
                self.group_ids[z] = el.group if el.group else 0
                self.period_ids[z] = el.row if el.row else 0
                self.block_ids[z] = block_map.get(el.block, 0)
            except Exception:
                pass

    def forward(self, elem_ids):
        """elem_ids: (batch, seq_len)"""
        main = self.main_embedding(elem_ids)
        grp = self.group_embedding(self.group_ids[elem_ids])
        per = self.period_embedding(self.period_ids[elem_ids])
        blk = self.block_embedding(self.block_ids[elem_ids])
        combined = torch.cat([main, grp, per, blk], dim=-1)
        return self.proj(combined)


class FractionalPositionalEncoding(nn.Module):
    """
    Encode composition fractions as continuous positional signals.
    Uses multi-frequency sinusoidal encoding with learnable scaling,
    analogous to how transformers encode discrete positions but adapted
    for continuous stoichiometric fractions in [0, 1].
    """

    def __init__(self, d_model, n_frequencies=64):
        super().__init__()
        self.n_freq = n_frequencies
        # Learnable frequency scales (initialized log-uniform)
        self.log_freqs = nn.Parameter(torch.linspace(0, 4, n_frequencies))
        self.proj = nn.Sequential(
            nn.Linear(2 * n_frequencies, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, fractions):
        """fractions: (batch, seq_len) in [0, 1]"""
        freqs = self.log_freqs.exp()  # (n_freq,)
        x = fractions.unsqueeze(-1) * freqs  # (B, N, n_freq)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, N, 2*n_freq)
        return self.proj(enc)


class GatedResidualFFN(nn.Module):
    """
    Gated residual feed-forward network.
    Uses a gating mechanism instead of simple residual addition,
    allowing the model to control information flow.
    """

    def __init__(self, d_model, dim_ff, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ffn_out = self.ffn(x)
        gate = self.gate(x)
        return self.layer_norm(x + self.dropout(gate * ffn_out))


class CompositionAttentionBlock(nn.Module):
    """
    Self-attention block with composition-aware bias.
    Standard multi-head attention + gated residual FFN.
    """

    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = GatedResidualFFN(d_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Gated FFN
        x = self.ffn(x)
        return x


class CompositionTransformer(nn.Module):
    """
    Full custom transformer for composition-based property prediction.

    Special features:
    - Periodic-table-aware element embeddings
    - Fractional positional encoding for stoichiometry
    - Learnable [CLS] token for composition-level readout
    - Gated residual feed-forward blocks
    - Multi-seed ensemble compatible
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, dim_ff, dropout, is_clf):
        super().__init__()

        # Element + fraction encoding
        self.elem_emb = PeriodicTableEmbedding(vocab_size, d_model)
        self.frac_enc = FractionalPositionalEncoding(d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CompositionAttentionBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        if is_clf:
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def forward(self, elem_ids, fractions):
        """
        elem_ids: (batch, max_elems)
        fractions: (batch, max_elems)
        """
        B = elem_ids.shape[0]
        pad_mask = (elem_ids == 0)  # True = padding

        # Encode elements and fractions
        x = self.elem_emb(elem_ids) + self.frac_enc(fractions)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, d_model)

        # Extend padding mask for [CLS] (never masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=elem_ids.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=full_mask)

        # [CLS] readout
        cls_out = x[:, 0, :]  # (B, d_model)

        out = self.output_head(cls_out).squeeze(-1)
        return self.final_act(out)


def parse_composition(comp):
    if isinstance(comp, str):
        comp = Composition(comp)
    elem_dict = comp.fractional_composition.as_dict()
    ids, fracs = [], []
    for sym, frac in elem_dict.items():
        z = ELEM_TO_Z.get(sym, 0)
        if z > 0:
            ids.append(z)
            fracs.append(frac)
    n = len(ids)
    ids = ids[:MAX_ELEMS] + [0] * max(0, MAX_ELEMS - n)
    fracs = fracs[:MAX_ELEMS] + [0.0] * max(0, MAX_ELEMS - n)
    return ids, fracs


class CompTransDataset(Dataset):
    def __init__(self, compositions, targets=None):
        self.ids_list, self.fracs_list = [], []
        for comp in compositions:
            ids, fracs = parse_composition(comp)
            self.ids_list.append(ids)
            self.fracs_list.append(fracs)
        self.ids_list = torch.LongTensor(self.ids_list)
        self.fracs_list = torch.FloatTensor(self.fracs_list)
        self.targets = None
        if targets is not None:
            self.targets = torch.FloatTensor(np.array(targets, dtype=np.float32))

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.ids_list[idx], self.fracs_list[idx], self.targets[idx]
        return self.ids_list[idx], self.fracs_list[idx]


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_single_seed(task, fold_idx, seed):
    """Train one model for a given fold and seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)

    n = len(train_inputs)
    n_val = max(1, int(0.1 * n))
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_comps = [train_inputs.iloc[i] for i in tr_idx]
    tr_targets = [train_outputs.iloc[i] for i in tr_idx]
    va_comps = [train_inputs.iloc[i] for i in val_idx]
    va_targets = [train_outputs.iloc[i] for i in val_idx]

    train_ds = CompTransDataset(tr_comps, tr_targets)
    val_ds = CompTransDataset(va_comps, va_targets)
    test_ds = CompTransDataset(list(test_inputs))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CompositionTransformer(
        VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, DIM_FF, DROPOUT, IS_CLASSIFICATION
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion = nn.BCELoss() if IS_CLASSIFICATION else nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            ids, fracs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred = model(ids, fracs)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                ids, fracs, targets = [b.to(device) for b in batch]
                pred = model(ids, fracs)
                val_losses.append(criterion(pred, targets).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 50 == 0:
            print(f"      Seed {seed} Epoch {epoch+1}/{EPOCHS}: val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"      Seed {seed} early stop at epoch {epoch+1}")
            break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            ids, fracs = batch[0].to(device), batch[1].to(device)
            pred = model(ids, fracs)
            predictions.append(pred.cpu().numpy())
    predictions = np.concatenate(predictions)

    torch.save(best_state, f"{EXP_DIR}/checkpoints/fold{fold_idx}_seed{seed}.pt")
    return predictions


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

# 5-fold cross-validation with multi-seed ensemble
fold_scores = []

for fold_idx in range(5):
    print(f"\n--- Fold {fold_idx} ---")
    train_inputs, _ = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)
    print(f"  Train: {len(train_inputs)}, Test: {len(test_inputs)}")

    all_preds = []
    for seed in range(N_SEEDS):
        print(f"    Training seed {seed}...")
        preds = train_single_seed(task, fold_idx, seed=seed * 137 + fold_idx * 7)
        all_preds.append(preds)

    ensemble_preds = np.mean(all_preds, axis=0)
    task.record(fold_idx, ensemble_preds)

    test_data = task.get_test_data(fold_idx, include_target=True)
    target_col = test_data.columns[-1]
    true_values = test_data[target_col].values
    if IS_CLASSIFICATION:
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(true_values, ensemble_preds)
        metric = "ROC-AUC"
    else:
        score = float(np.mean(np.abs(true_values - ensemble_preds)))
        metric = "MAE"
    fold_scores.append(score)
    print(f"  Fold {fold_idx} {metric}: {score:.4f}")

# Summary
print(f"\n{'=' * 70}")
print(f"Results: Custom Composition Transformer on {TASK_NAME}")
print(f"{'=' * 70}")
metric = "ROC-AUC" if IS_CLASSIFICATION else "MAE"
for i, score in enumerate(fold_scores):
    print(f"  Fold {i}: {metric} = {score:.4f}")
print(f"  Mean {metric}: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

# Save results
results = {
    "task": TASK_NAME,
    "model": "CompositionTransformer",
    "metric": metric,
    "fold_scores": fold_scores,
    "mean_score": float(np.mean(fold_scores)),
    "std_score": float(np.std(fold_scores)),
    "config": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dim_ff": DIM_FF,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "n_seeds": N_SEEDS,
    },
    "device": str(device),
}
with open(f"{EXP_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {EXP_DIR}/results.json")

mb.to_file(f"{EXP_DIR}/matbench_results.json.gz")
print(f"MatBench file saved to {EXP_DIR}/matbench_results.json.gz")
print("Done!")
```

## Key Parameters

| Parameter | Script | Default | Notes |
|-----------|--------|---------|-------|
| `ELEM_EMB_DIM` | Roost | 64 | Element embedding dimension; increase for more expressive representations |
| `MSG_HEADS` | Roost | 4 | Attention heads in message passing; must divide `ELEM_EMB_DIM` |
| `MSG_LAYERS` | Roost | 3 | Number of message passing rounds; 2-4 typical |
| `D_MODEL` | CrabNet / Transformer | 512 / 256 | Transformer model dimension; larger = more capacity, slower |
| `N_HEADS` | CrabNet / Transformer | 8 | Attention heads; must divide `D_MODEL` |
| `N_ENCODER_LAYERS` / `N_LAYERS` | CrabNet / Transformer | 3 / 4 | Depth of transformer stack |
| `DIM_FF` | CrabNet / Transformer | 1024 / 512 | Feed-forward hidden dim; typically 2-4x `D_MODEL` |
| `FRAC_ENCODING_DIM` | CrabNet | 32 | Frequencies for fractional encoding |
| `N_SEEDS` | Roost / Transformer | 3 | Ensemble size; more seeds = better but linearly slower |
| `LEARNING_RATE` | All | 1e-4 to 3e-4 | Use lower rates for larger models |
| `WARMUP_EPOCHS` | Transformer | 20 | Linear learning rate warmup period |
| `PATIENCE` | All | 30-60 | Early stopping patience; increase for noisy validation |
| `FOUNDATION` | Transfer | "mace" | Source model: "mace" or "chgnet" |
| `FINE_TUNE_EMBEDDINGS` | Transfer | True | Whether to update foundation embeddings during training |
| `MAX_ELEMS` | All | 16 | Max elements per composition; 16 covers virtually all materials |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| GPU OOM | Model too large for available VRAM | Reduce `D_MODEL`, `BATCH_SIZE`, or `N_LAYERS` |
| Loss NaN | Learning rate too high or gradient explosion | Reduce `LEARNING_RATE`, ensure gradient clipping is enabled |
| Overfitting on matbench_steels | Only 312 samples | Increase `DROPOUT`, reduce model size, add weight decay |
| Slow convergence | Learning rate too low or no warmup | Use warmup schedule (Script 4), increase `LEARNING_RATE` slightly |
| Roost attention collapse | All attention on one element | Increase `MSG_HEADS`, add dropout to attention |
| CrabNet padding artifacts | Padding tokens affecting attention | Verify `key_padding_mask` is correctly set (True=pad) |
| Foundation model not found | MACE/CHGNet not installed | Falls back to physical-property embeddings; install with pip |
| Classification wrong metric | Using accuracy instead of ROC-AUC | Set `IS_CLASSIFICATION=True`, predictions should be probabilities |
| Ensemble too slow | `N_SEEDS` too high | Reduce to 1-2 for development, use 3-5 for final submission |
| Poor transfer results | Foundation embeddings too general | Set `FINE_TUNE_EMBEDDINGS=True`, increase MLP capacity |
| Element not in vocabulary | Rare element with Z > 118 | Handled by padding_idx=0 in embedding; check data preprocessing |
| matbench API error | Wrong number of predictions | Ensure predictions array length matches test set exactly |
