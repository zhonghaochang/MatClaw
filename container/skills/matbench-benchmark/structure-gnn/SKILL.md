# Build Structure-Based GNN Models for MatBench

## ⚠️ A100 GPU Optimization — READ FIRST

**Before writing ANY training code, read the "A100 GPU Optimization" section in `~/.claude/skills/matbench-benchmark/training-pipeline/SKILL.md`.** Previous runs only used 5-8% of A100 80GB VRAM due to small batch sizes and unoptimized DataLoaders.

**Mandatory for ALL GNN DataLoaders:**
```python
loader = DataLoader(dataset, batch_size=BS, shuffle=True,
                    num_workers=4, pin_memory=True,
                    persistent_workers=True, prefetch_factor=2)
```
**Mandatory A100 settings** (put at top of every script):
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

## When to Use

- The MatBench task input is a **pymatgen Structure** (8 of the 13 tasks)
- You want graph neural networks that learn directly from crystal structure
- You need CGCNN, SchNet, or DimeNet++ architectures
- You want to build custom GNN architectures using `torch_geometric.nn.MessagePassing`

### Structure-Input Tasks

| Task | Samples | Property |
|------|---------|----------|
| matbench_jdft2d | 636 | Exfoliation energy (meV/atom) |
| matbench_phonons | 1,265 | Max phonon frequency (1/cm) |
| matbench_log_gvrh | 10,987 | log10 bulk modulus (GPa) |
| matbench_log_kvrh | 10,987 | log10 shear modulus (GPa) |
| matbench_perovskites | 18,928 | Formation energy (eV) |
| matbench_mp_gap | 106,113 | Band gap (eV) |
| matbench_mp_is_metal | 106,113 | Metal / non-metal |
| matbench_mp_e_form | 132,752 | Formation energy (eV/atom) |

## Method Selection

```
Start here: Which GNN architecture?
│
├─ First attempt / baseline?
│  └─ → CGCNN (Script 2)
│     Fast training, good accuracy, easiest to tune
│
├─ Want better accuracy, OK with slower training?
│  ├─ Dataset > 10k samples?
│  │  └─ → SchNet (Script 3)
│  │     Continuous filter convolutions, good scalability
│  │
│  └─ Dataset < 10k samples?
│     └─ → DimeNet++ (Script 4)
│        Best on small structure tasks, captures angles
│
├─ Want to experiment with custom message passing?
│  └─ → Custom GNN Template (Script 5)
│     Minimal MessagePassing scaffold to modify
│
└─ Unsure about graph construction?
   └─ → Crystal Graph Construction (Script 1)
      Understand radius, neighbors, edge features first
```

## Prerequisites

- **Python:** `/opt/conda/envs/matbench/bin/python`
- **Packages:** torch (CUDA 12.8), torch-geometric, e3nn, pymatgen, matbench, numpy
- **GPU:** A100-SXM4-80GB (80 GB VRAM, CUDA 12.8)
- **Storage:** `/workspace/group/matbench/` (data disk)

## Detailed Steps

---

### Script 1: Crystal Graph Construction

Convert pymatgen `Structure` objects to `torch_geometric.data.Data` graph objects with caching.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Crystal Graph Construction: Convert pymatgen Structure to torch-geometric Data.
Builds atom node features (one-hot Z), edge connectivity (radius neighbors),
and Gaussian RBF edge features. Results are cached to disk.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure

# ── Settings ──────────────────────────────────────────────────────────────────
TASK_NAME = "matbench_mp_e_form"          # Change per task
RADIUS = 8.0                              # Angstrom cutoff
MAX_NEIGHBORS = 12                        # Max neighbors per atom
NUM_RBF = 50                              # Gaussian RBF centers
CACHE_DIR = f"/workspace/group/matbench/data/graphs/{TASK_NAME}"

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Graph construction ────────────────────────────────────────────────────────
def structure_to_graph(structure, radius=RADIUS, max_neighbors=MAX_NEIGHBORS,
                       num_rbf=NUM_RBF, target=None):
    """Convert pymatgen Structure to torch-geometric Data object.

    Node features: one-hot atomic number (max 100 elements)
    Edge features: Gaussian RBF expansion of pairwise distance
    """
    # Atom features: one-hot atomic number (Z capped at 100)
    atom_features = []
    for site in structure:
        z = min(site.specie.Z, 100)
        feat = [0.0] * 100
        feat[z - 1] = 1.0
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)

    # Get all neighbors within radius
    all_neighbors = structure.get_all_neighbors(radius)

    # Build edge list and Gaussian RBF edge attributes
    edge_src, edge_dst = [], []
    edge_attr_list = []
    centers = np.linspace(0, radius, num_rbf)
    gamma = 0.5

    for i, neighbors in enumerate(all_neighbors):
        # Sort by distance, keep closest max_neighbors
        sorted_neighbors = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for neighbor in sorted_neighbors:
            j = neighbor[2]       # neighbor site index
            dist = neighbor[1]    # distance in Angstroms
            edge_src.append(i)
            edge_dst.append(j)
            rbf = np.exp(-gamma * (dist - centers) ** 2)
            edge_attr_list.append(rbf.tolist())

    if len(edge_src) == 0:
        # Fallback: self-loop if no neighbors found
        edge_src, edge_dst = [0], [0]
        edge_attr_list = [np.zeros(num_rbf).tolist()]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data


# ── Caching helpers ───────────────────────────────────────────────────────────
def _cache_path(idx):
    return os.path.join(CACHE_DIR, f"graph_{idx:06d}.pt")


def build_graph_dataset(structures, targets=None, desc="structures"):
    """Build graphs for a list of structures. Uses disk cache."""
    graphs = []
    cached, built = 0, 0
    total = len(structures)

    for idx in range(total):
        cp = _cache_path(idx)
        if os.path.exists(cp):
            data = torch.load(cp, weights_only=False)
            # Update target if provided (fold-specific)
            if targets is not None:
                data.y = torch.tensor([targets[idx]], dtype=torch.float)
            graphs.append(data)
            cached += 1
        else:
            t = targets[idx] if targets is not None else None
            data = structure_to_graph(structures[idx], target=t)
            torch.save(data, cp)
            graphs.append(data)
            built += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"  [{desc}] {idx+1}/{total} "
                  f"(cached={cached}, built={built})")

    print(f"  Done: {cached} loaded from cache, {built} newly built")
    return graphs


# ── Demo: process one fold of a task ─────────────────────────────────────────
if __name__ == "__main__":
    os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, TASK_NAME)
    task.load()

    print(f"Task: {TASK_NAME}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Radius: {RADIUS} A, Max neighbors: {MAX_NEIGHBORS}, RBF centers: {NUM_RBF}")
    print()

    # Process fold 0 as a demo
    fold = 0
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    print(f"Fold {fold}: {len(train_inputs)} training structures")

    t0 = time.time()
    graphs = build_graph_dataset(
        train_inputs.tolist(), train_outputs.tolist(), desc=f"fold{fold}"
    )
    elapsed = time.time() - t0

    print(f"\nBuilt {len(graphs)} graphs in {elapsed:.1f}s")
    print(f"Sample graph: {graphs[0]}")
    print(f"  Nodes: {graphs[0].x.shape[0]}, Edges: {graphs[0].edge_index.shape[1]}")
    print(f"  Node feature dim: {graphs[0].x.shape[1]}")
    print(f"  Edge feature dim: {graphs[0].edge_attr.shape[1]}")
    print(f"  Target: {graphs[0].y.item():.4f}")
    print("\nDone.")
```

---

### Script 2: CGCNN -- Complete MatBench Pipeline

Full CGCNN implementation using `torch_geometric.nn.CGConv` with the standard 5-fold MatBench protocol.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
CGCNN on MatBench: Crystal Graph Convolutional Neural Network.
Full 5-fold benchmark pipeline with caching, AMP, early stopping.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core import Structure

os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
TASK_NAME = "matbench_mp_e_form"        # Change to target task
RADIUS = 8.0
MAX_NEIGHBORS = 12
NUM_RBF = 50
HIDDEN_DIM = 64
NUM_CONV = 3
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 300
PATIENCE = 50
SAVE_DIR = "/workspace/group/matbench/models"
CACHE_DIR = f"/workspace/group/matbench/data/graphs/cgcnn_{TASK_NAME}"
PLOT_DIR = "/workspace/group/matbench/plots"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ── Graph construction (same as Script 1) ────────────────────────────────────
def structure_to_graph(structure, radius=RADIUS, max_neighbors=MAX_NEIGHBORS,
                       num_rbf=NUM_RBF, target=None):
    atom_features = []
    for site in structure:
        z = min(site.specie.Z, 100)
        feat = [0.0] * 100
        feat[z - 1] = 1.0
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)

    all_neighbors = structure.get_all_neighbors(radius)
    edge_src, edge_dst, edge_attr_list = [], [], []
    centers = np.linspace(0, radius, num_rbf)
    gamma = 0.5

    for i, neighbors in enumerate(all_neighbors):
        sorted_nbrs = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for nbr in sorted_nbrs:
            j, dist = nbr[2], nbr[1]
            edge_src.append(i)
            edge_dst.append(j)
            rbf = np.exp(-gamma * (dist - centers) ** 2)
            edge_attr_list.append(rbf.tolist())

    if len(edge_src) == 0:
        edge_src, edge_dst = [0], [0]
        edge_attr_list = [np.zeros(num_rbf).tolist()]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data


def _cache_path(idx):
    return os.path.join(CACHE_DIR, f"graph_{idx:06d}.pt")


def build_graphs(structures, targets, desc="graphs"):
    graphs = []
    total = len(structures)
    for idx in range(total):
        cp = _cache_path(idx)
        if os.path.exists(cp):
            data = torch.load(cp, weights_only=False)
            data.y = torch.tensor([targets[idx]], dtype=torch.float)
        else:
            data = structure_to_graph(structures[idx], target=targets[idx])
            torch.save(data, cp)
        graphs.append(data)
        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"  [{desc}] {idx+1}/{total}")
    return graphs


# ── CGCNN Model ───────────────────────────────────────────────────────────────
class CGCNN(nn.Module):
    def __init__(self, atom_fea_dim=100, edge_fea_dim=50, hidden_dim=64,
                 n_conv=3, out_dim=1):
        super().__init__()
        self.atom_embedding = nn.Linear(atom_fea_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_fea_dim, batch_norm=True)
            for _ in range(n_conv)
        ])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = self.relu(self.atom_embedding(x))
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
            loss = criterion(pred, batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
        preds.append(pred.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = np.mean(np.abs(preds - trues))
    return mae, preds


# ── Main: 5-fold MatBench benchmark ──────────────────────────────────────────
if __name__ == "__main__":
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, TASK_NAME)
    task.load()

    print(f"{'='*60}")
    print(f"CGCNN on {TASK_NAME}")
    print(f"Hidden={HIDDEN_DIM}, Conv={NUM_CONV}, LR={LR}, BS={BATCH_SIZE}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}")
    print(f"{'='*60}\n")

    fold_maes = []

    for fold in task.folds:
        print(f"\n{'─'*40} Fold {fold} {'─'*40}")

        # Load data
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        structures_train = train_inputs.tolist()
        targets_train = train_outputs.tolist()
        structures_test = test_inputs.tolist()
        targets_test = test_outputs.tolist()

        # Build graphs
        print("Building training graphs...")
        train_graphs = build_graphs(structures_train, targets_train,
                                    desc=f"train_fold{fold}")
        print("Building test graphs...")
        test_graphs = build_graphs(structures_test, targets_test,
                                   desc=f"test_fold{fold}")

        # Data loaders
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4,
                                  pin_memory=True)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True)

        # Model, optimizer, scheduler
        model = CGCNN(
            atom_fea_dim=100, edge_fea_dim=NUM_RBF,
            hidden_dim=HIDDEN_DIM, n_conv=NUM_CONV
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=(device.type == "cuda"))

        # Training loop with early stopping
        best_mae = float("inf")
        patience_counter = 0
        best_state = None
        ckpt_path = os.path.join(
            SAVE_DIR, f"cgcnn_{TASK_NAME}_fold{fold}_best.pt"
        )

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device
            )
            test_mae, _ = evaluate(model, test_loader, device)
            scheduler.step()
            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                      f"Loss={train_loss:.5f} | MAE={test_mae:.5f} | "
                      f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                      f"{elapsed:.1f}s")

            if test_mae < best_mae:
                best_mae = test_mae
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in
                              model.state_dict().items()}
                torch.save(best_state, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best MAE={best_mae:.5f})")
                    break

        # Load best model and get final predictions
        model.load_state_dict(torch.load(ckpt_path, weights_only=False))
        model.to(device)
        final_mae, test_preds = evaluate(model, test_loader, device)
        fold_maes.append(final_mae)

        # Record predictions for matbench
        task.record(fold, test_preds)

        print(f"  Fold {fold} best MAE: {final_mae:.5f}")
        print(f"  Checkpoint: {ckpt_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"CGCNN {TASK_NAME} Results:")
    for i, mae in enumerate(fold_maes):
        print(f"  Fold {i}: MAE = {mae:.5f}")
    print(f"  Mean MAE: {np.mean(fold_maes):.5f} +/- {np.std(fold_maes):.5f}")
    print(f"{'='*60}")

    # Save benchmark results
    results_dir = "/workspace/group/matbench/results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"cgcnn_{TASK_NAME}.json")
    results = {
        "model": "CGCNN",
        "task": TASK_NAME,
        "fold_maes": fold_maes,
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes)),
        "config": {
            "radius": RADIUS, "max_neighbors": MAX_NEIGHBORS,
            "hidden_dim": HIDDEN_DIM, "n_conv": NUM_CONV,
            "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS,
        }
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    print("\nDone.")
```

---

### Script 3: SchNet -- Complete MatBench Pipeline

SchNet uses continuous filter convolutions on atom positions and atomic numbers. Graph construction differs from CGCNN: it needs Cartesian coordinates (`pos`) and atomic numbers (`z`) instead of one-hot node features.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
SchNet on MatBench: Continuous-filter convolutional neural network.
Uses atomic positions and numbers directly. Full 5-fold benchmark.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SchNet as SchNetModel
from pymatgen.core import Structure

os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
TASK_NAME = "matbench_mp_e_form"
RADIUS = 6.0
MAX_NEIGHBORS = 32
HIDDEN_CHANNELS = 128
NUM_FILTERS = 128
NUM_INTERACTIONS = 6
NUM_GAUSSIANS = 50
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 300
PATIENCE = 50
SAVE_DIR = "/workspace/group/matbench/models"
CACHE_DIR = f"/workspace/group/matbench/data/graphs/schnet_{TASK_NAME}"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Graph construction for SchNet ─────────────────────────────────────────────
def structure_to_schnet_graph(structure, radius=RADIUS,
                              max_neighbors=MAX_NEIGHBORS, target=None):
    """Build graph with Cartesian positions and atomic numbers for SchNet.

    SchNet needs:
      - z: atomic numbers (long tensor)
      - pos: Cartesian coordinates (float tensor)
      - edge_index: neighbor pairs within cutoff
    """
    # Atomic numbers
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # Cartesian coordinates from fractional
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    # Build neighbor list
    all_neighbors = structure.get_all_neighbors(radius)
    edge_src, edge_dst = [], []
    for i, neighbors in enumerate(all_neighbors):
        sorted_nbrs = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for nbr in sorted_nbrs:
            j = nbr[2]
            edge_src.append(i)
            edge_dst.append(j)

    if len(edge_src) == 0:
        edge_src, edge_dst = [0], [0]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    data = Data(z=z, pos=pos, edge_index=edge_index)
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data


def _cache_path(idx):
    return os.path.join(CACHE_DIR, f"graph_{idx:06d}.pt")


def build_graphs(structures, targets, desc="graphs"):
    graphs = []
    total = len(structures)
    for idx in range(total):
        cp = _cache_path(idx)
        if os.path.exists(cp):
            data = torch.load(cp, weights_only=False)
            data.y = torch.tensor([targets[idx]], dtype=torch.float)
        else:
            data = structure_to_schnet_graph(structures[idx], target=targets[idx])
            torch.save(data, cp)
        graphs.append(data)
        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"  [{desc}] {idx+1}/{total}")
    return graphs


# ── SchNet wrapper (handles batch format) ─────────────────────────────────────
class SchNetWrapper(nn.Module):
    """Wraps torch_geometric SchNet to accept Data batch objects."""

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=6.0):
        super().__init__()
        self.schnet = SchNetModel(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )

    def forward(self, data):
        # SchNet expects z, pos, batch
        out = self.schnet(data.z, data.pos, data.batch)
        return out.squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
            loss = criterion(pred, batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
        preds.append(pred.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = np.mean(np.abs(preds - trues))
    return mae, preds


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, TASK_NAME)
    task.load()

    print(f"{'='*60}")
    print(f"SchNet on {TASK_NAME}")
    print(f"Hidden={HIDDEN_CHANNELS}, Interactions={NUM_INTERACTIONS}")
    print(f"Cutoff={RADIUS}, Gaussians={NUM_GAUSSIANS}")
    print(f"LR={LR}, BS={BATCH_SIZE}, Epochs={EPOCHS}")
    print(f"{'='*60}\n")

    fold_maes = []

    for fold in task.folds:
        print(f"\n{'─'*40} Fold {fold} {'─'*40}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        print("Building training graphs...")
        train_graphs = build_graphs(
            train_inputs.tolist(), train_outputs.tolist(),
            desc=f"train_fold{fold}"
        )
        print("Building test graphs...")
        test_graphs = build_graphs(
            test_inputs.tolist(), targets_test := test_outputs.tolist(),
            desc=f"test_fold{fold}"
        )

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4, pin_memory=True)

        model = SchNetWrapper(
            hidden_channels=HIDDEN_CHANNELS,
            num_filters=NUM_FILTERS,
            num_interactions=NUM_INTERACTIONS,
            num_gaussians=NUM_GAUSSIANS,
            cutoff=RADIUS,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=(device.type == "cuda"))

        best_mae = float("inf")
        patience_counter = 0
        ckpt_path = os.path.join(
            SAVE_DIR, f"schnet_{TASK_NAME}_fold{fold}_best.pt"
        )

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device
            )
            test_mae, _ = evaluate(model, test_loader, device)
            scheduler.step()
            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                      f"Loss={train_loss:.5f} | MAE={test_mae:.5f} | "
                      f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                      f"{elapsed:.1f}s")

            if test_mae < best_mae:
                best_mae = test_mae
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best MAE={best_mae:.5f})")
                    break

        model.load_state_dict(torch.load(ckpt_path, weights_only=False))
        model.to(device)
        final_mae, test_preds = evaluate(model, test_loader, device)
        fold_maes.append(final_mae)

        task.record(fold, test_preds)
        print(f"  Fold {fold} best MAE: {final_mae:.5f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SchNet {TASK_NAME} Results:")
    for i, mae in enumerate(fold_maes):
        print(f"  Fold {i}: MAE = {mae:.5f}")
    print(f"  Mean MAE: {np.mean(fold_maes):.5f} +/- {np.std(fold_maes):.5f}")
    print(f"{'='*60}")

    results_dir = "/workspace/group/matbench/results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"schnet_{TASK_NAME}.json")
    results = {
        "model": "SchNet", "task": TASK_NAME,
        "fold_maes": fold_maes,
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes)),
        "config": {
            "hidden_channels": HIDDEN_CHANNELS,
            "num_interactions": NUM_INTERACTIONS,
            "cutoff": RADIUS, "num_gaussians": NUM_GAUSSIANS,
            "batch_size": BATCH_SIZE, "lr": LR,
        }
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    print("\nDone.")
```

---

### Script 4: DimeNet++ -- Complete MatBench Pipeline

DimeNet++ captures angular information between triplets of atoms. Most accurate for small-to-medium structure tasks but slower and more memory-intensive.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
DimeNet++ on MatBench: Directional message passing with angular information.
Best accuracy on small structure tasks. Full 5-fold benchmark.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from pymatgen.core import Structure

os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
TASK_NAME = "matbench_perovskites"      # Good for DimeNet++ (medium size)
RADIUS = 5.0
MAX_NEIGHBORS = 20
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 1
NUM_BLOCKS = 4
INT_EMB_SIZE = 64
BASIS_EMB_SIZE = 8
OUT_EMB_CHANNELS = 256
NUM_SPHERICAL = 7
NUM_RADIAL = 6
BATCH_SIZE = 32                         # Smaller due to memory
LR = 1e-4
EPOCHS = 300
PATIENCE = 50
SAVE_DIR = "/workspace/group/matbench/models"
CACHE_DIR = f"/workspace/group/matbench/data/graphs/dimenet_{TASK_NAME}"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ── Graph construction for DimeNet++ ──────────────────────────────────────────
def structure_to_dimenet_graph(structure, radius=RADIUS,
                               max_neighbors=MAX_NEIGHBORS, target=None):
    """Build graph with positions and atomic numbers for DimeNet++.

    DimeNet++ needs z (atomic numbers), pos (Cartesian coords).
    It builds its own edge list internally from pos, but we pre-build
    edge_index for consistency with the DataLoader.
    """
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    all_neighbors = structure.get_all_neighbors(radius)
    edge_src, edge_dst = [], []
    for i, neighbors in enumerate(all_neighbors):
        sorted_nbrs = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for nbr in sorted_nbrs:
            j = nbr[2]
            edge_src.append(i)
            edge_dst.append(j)

    if len(edge_src) == 0:
        edge_src, edge_dst = [0], [0]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    data = Data(z=z, pos=pos, edge_index=edge_index)
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data


def _cache_path(idx):
    return os.path.join(CACHE_DIR, f"graph_{idx:06d}.pt")


def build_graphs(structures, targets, desc="graphs"):
    graphs = []
    total = len(structures)
    for idx in range(total):
        cp = _cache_path(idx)
        if os.path.exists(cp):
            data = torch.load(cp, weights_only=False)
            data.y = torch.tensor([targets[idx]], dtype=torch.float)
        else:
            data = structure_to_dimenet_graph(
                structures[idx], target=targets[idx]
            )
            torch.save(data, cp)
        graphs.append(data)
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"  [{desc}] {idx+1}/{total}")
    return graphs


# ── DimeNet++ wrapper ─────────────────────────────────────────────────────────
class DimeNetPPWrapper(nn.Module):
    """Wraps DimeNetPlusPlus for batch Data objects."""

    def __init__(self, hidden_channels=128, out_channels=1, num_blocks=4,
                 int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, cutoff=5.0):
        super().__init__()
        self.model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
        )

    def forward(self, data):
        out = self.model(data.z, data.pos, data.batch)
        return out.squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
            loss = criterion(pred, batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(batch)
        preds.append(pred.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = np.mean(np.abs(preds - trues))
    return mae, preds


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, TASK_NAME)
    task.load()

    print(f"{'='*60}")
    print(f"DimeNet++ on {TASK_NAME}")
    print(f"Hidden={HIDDEN_CHANNELS}, Blocks={NUM_BLOCKS}")
    print(f"Spherical={NUM_SPHERICAL}, Radial={NUM_RADIAL}")
    print(f"Cutoff={RADIUS}, LR={LR}, BS={BATCH_SIZE}")
    print(f"{'='*60}\n")

    fold_maes = []

    for fold in task.folds:
        print(f"\n{'─'*40} Fold {fold} {'─'*40}")

        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        print("Building training graphs...")
        train_graphs = build_graphs(
            train_inputs.tolist(), train_outputs.tolist(),
            desc=f"train_fold{fold}"
        )
        print("Building test graphs...")
        test_graphs = build_graphs(
            test_inputs.tolist(), test_outputs.tolist(),
            desc=f"test_fold{fold}"
        )

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4, pin_memory=True)

        model = DimeNetPPWrapper(
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            num_blocks=NUM_BLOCKS,
            int_emb_size=INT_EMB_SIZE,
            basis_emb_size=BASIS_EMB_SIZE,
            out_emb_channels=OUT_EMB_CHANNELS,
            num_spherical=NUM_SPHERICAL,
            num_radial=NUM_RADIAL,
            cutoff=RADIUS,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=(device.type == "cuda"))

        best_mae = float("inf")
        patience_counter = 0
        ckpt_path = os.path.join(
            SAVE_DIR, f"dimenetpp_{TASK_NAME}_fold{fold}_best.pt"
        )

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device
            )
            test_mae, _ = evaluate(model, test_loader, device)
            scheduler.step()
            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                      f"Loss={train_loss:.5f} | MAE={test_mae:.5f} | "
                      f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                      f"{elapsed:.1f}s")

            if test_mae < best_mae:
                best_mae = test_mae
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best MAE={best_mae:.5f})")
                    break

        model.load_state_dict(torch.load(ckpt_path, weights_only=False))
        model.to(device)
        final_mae, test_preds = evaluate(model, test_loader, device)
        fold_maes.append(final_mae)

        task.record(fold, test_preds)
        print(f"  Fold {fold} best MAE: {final_mae:.5f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DimeNet++ {TASK_NAME} Results:")
    for i, mae in enumerate(fold_maes):
        print(f"  Fold {i}: MAE = {mae:.5f}")
    print(f"  Mean MAE: {np.mean(fold_maes):.5f} +/- {np.std(fold_maes):.5f}")
    print(f"{'='*60}")

    results_dir = "/workspace/group/matbench/results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"dimenetpp_{TASK_NAME}.json")
    results = {
        "model": "DimeNet++", "task": TASK_NAME,
        "fold_maes": fold_maes,
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes)),
        "config": {
            "hidden_channels": HIDDEN_CHANNELS,
            "num_blocks": NUM_BLOCKS, "cutoff": RADIUS,
            "num_spherical": NUM_SPHERICAL, "num_radial": NUM_RADIAL,
            "batch_size": BATCH_SIZE, "lr": LR,
        }
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    print("\nDone.")
```

---

### Script 5: Custom GNN Template

Minimal `MessagePassing`-based GNN for users who want to design their own architecture.

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""
Custom GNN Template: Build your own message-passing neural network.
Inherit from torch_geometric.nn.MessagePassing and define message/update.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Custom Message Passing Layer ──────────────────────────────────────────────
class CustomConv(MessagePassing):
    """Custom graph convolution using message passing.

    Override these methods to define your own GNN:
      - message():    compute messages from neighbors
      - aggregate():  combine neighbor messages (sum, mean, max)
      - update():     update node embeddings with aggregated messages
    """

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr="mean")  # "mean", "sum", or "max"
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_update = nn.Linear(out_channels * 2, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        x = self.lin_node(x)
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Residual connection
        out = out + x
        out = self.norm(out)
        out = self.act(out)
        return out

    def message(self, x_j, edge_attr):
        """Compute message from neighbor j to node i.
        x_j: features of source node j
        edge_attr: edge features (e.g., RBF distance expansion)
        """
        edge_feat = self.lin_edge(edge_attr)
        return x_j * edge_feat  # element-wise multiply

    def update(self, aggr_out, x):
        """Update node embedding after aggregation.
        aggr_out: aggregated messages from neighbors
        x: current node features
        """
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.lin_update(combined)


# ── Custom GNN Model ──────────────────────────────────────────────────────────
class CustomGNN(nn.Module):
    """Complete GNN model using custom message passing layers.

    Modify this class to experiment with:
      - Different number/type of conv layers
      - Different readout functions (mean, sum, attention)
      - Different MLP head architectures
    """

    def __init__(self, atom_fea_dim=100, edge_fea_dim=50, hidden_dim=64,
                 n_conv=3, out_dim=1, dropout=0.1):
        super().__init__()
        self.atom_embed = nn.Linear(atom_fea_dim, hidden_dim)
        self.convs = nn.ModuleList([
            CustomConv(hidden_dim, hidden_dim, edge_fea_dim)
            for _ in range(n_conv)
        ])

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        # Embed atoms
        x = self.atom_embed(x)

        # Message passing layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # Graph-level readout (mean pool)
        x = global_mean_pool(x, batch)

        # Predict property
        return self.readout(x).squeeze(-1)


# ── Demo usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from matbench.bench import MatbenchBenchmark

    # Build a small test to verify the model runs
    print("CustomGNN Template - Verification Run")
    print(f"Device: {device}")

    # Create a dummy graph
    num_atoms = 10
    x = torch.randn(num_atoms, 100)
    edge_index = torch.randint(0, num_atoms, (2, 30))
    edge_attr = torch.randn(30, 50)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    y = torch.tensor([1.5])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                batch=batch, y=y)

    model = CustomGNN(atom_fea_dim=100, edge_fea_dim=50,
                      hidden_dim=64, n_conv=3).to(device)
    data = data.to(device)

    # Forward pass
    pred = model(data)
    print(f"Input: {num_atoms} atoms, 30 edges")
    print(f"Output: {pred.item():.4f} (target: {y.item():.4f})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Backward pass
    loss = nn.MSELoss()(pred, data.y)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradients OK: {all(p.grad is not None for p in model.parameters())}")

    print("\nTemplate verified. Modify CustomConv.message() and "
          "CustomConv.update() to experiment.")
    print("Use the graph construction from Script 1 to build real data.")
    print("\nDone.")
```

---

## Key Parameters

| Parameter | Description | CGCNN | SchNet | DimeNet++ |
|---|---|---|---|---|
| `radius_cutoff` | Neighbor search radius (angstrom) | 8.0 | 6.0 | 5.0 |
| `max_neighbors` | Max neighbors per atom | 12 | 32 | 20 |
| `hidden_channels` | Hidden layer width | 64 | 128 | 128 |
| `n_conv_layers` | Graph convolution layers | 3 | 6 | 4 |
| `batch_size` | Graphs per batch | 128 | 64 | 32 |
| `learning_rate` | Initial learning rate | 1e-3 | 5e-4 | 1e-4 |
| `pool` | Graph-level aggregation | mean | mean | mean |
| `edge_features` | Edge representation | Gaussian RBF | Gaussian filter | Spherical Bessel |
| `angle_info` | Uses bond angles | No | No | Yes |
| `training_speed` | Relative speed (A100) | Fast | Medium | Slow |

## Common Issues

| Problem | Cause | Solution |
|---|---|---|
| **CUDA out of memory** | Batch too large or graphs too dense | Reduce `batch_size` (try 32 or 16). Reduce `max_neighbors`. Use `torch.cuda.empty_cache()` between folds. |
| **Slow graph construction** | Recomputing graphs every run | Use the caching pattern: save `.pt` files to `CACHE_DIR` and check before building. First run is slow, subsequent runs load from disk. |
| **Model predicts constant value** | Targets not normalized, or LR too low | Normalize targets to zero mean / unit variance. Increase LR. Check that `data.y` is set correctly. |
| **NaN loss** | LR too high, exploding gradients, bad data | Reduce LR by 10x. Enable gradient clipping (`max_norm=10`). Check for NaN/Inf in input structures. |
| **Slow DataLoader** | `num_workers` too low or too high | Set `num_workers=4` with `pin_memory=True` for A100. Increase if CPU-bound, decrease if fork overhead dominates. |
| **Poor performance on small tasks** | Not enough data for deep GNN | Use fewer conv layers (2-3). Add dropout (0.1-0.2). Consider DimeNet++ which excels on small datasets. Try smaller `hidden_dim`. |
| **SchNet/DimeNet++ ignoring edges** | Edge index not used by model | These models build their own edges from `pos` internally. Ensure `pos` (Cartesian coords) and `z` (atomic numbers) are set correctly. |
| **Different results across runs** | Random initialization | Set seeds: `torch.manual_seed(42)`, `np.random.seed(42)`. Use `torch.backends.cudnn.deterministic = True`. |
