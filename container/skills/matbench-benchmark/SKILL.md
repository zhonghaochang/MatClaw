---
name: matbench-benchmark
description: MatBench Materials Science Benchmarking (9 sub-skills: task-setup, composition-models, structure-gnn, sota-reproduction, sota-reference, training-pipeline, evaluation-submission, model-optimization)
---

# MatBench Materials Science Benchmarking

MatBench is a standardized benchmark suite for evaluating machine learning models on materials science property prediction tasks. It contains 13 supervised learning tasks spanning composition-based and structure-based inputs, covering regression and classification problems. All tasks use a strict 5-fold nested cross-validation protocol with pre-defined splits, ensuring fair and reproducible comparison across models. MatBench is the de facto standard for reporting ML performance in computational materials science, with an official leaderboard tracking state-of-the-art results.

## Environment

All scripts **must** use the isolated MatBench conda environment:

```
/opt/conda/envs/matbench/bin/python
```

This is a Python 3.11 environment with the following packages installed:

| Package | Purpose |
|---------|---------|
| matbench | Benchmark framework, data loading, recording, submission |
| matminer | Featurization (Magpie, structural fingerprints, etc.) |
| scikit-learn | Traditional ML models (RF, GBR, SVM) |
| torch (CUDA 12.8) | PyTorch with GPU support |
| torch-geometric | Graph neural networks (CGCNN, SchNet, DimeNet++) |
| e3nn | Equivariant neural networks (MACE, NequIP) |
| tensorboard | Training visualization |
| pymatgen | Crystal structure manipulation |

**GPU:** NVIDIA A100-SXM4-80GB. Auto-detect with:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Important:** Always set `matplotlib.use("Agg")` before importing `pyplot`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

## File Storage — Experiment Management

All data, models, results, and outputs go to `/workspace/group/matbench/` (data disk, not system disk).

**CRITICAL RULES:**
1. Every training run MUST go into a unique experiment directory. NEVER write directly to a flat `models/` or `results/` directory. NEVER overwrite files from a previous experiment.
2. The `data/` directory contains **shared dataset cache** used by ALL environments and experiments. **NEVER delete, move, or overwrite files in `data/`**. New datasets will auto-download alongside existing ones.
3. When creating isolated conda environments for SOTA models, always install `matbench matminer` in that env too, and set `MATBENCH_DATA_HOME=/workspace/group/matbench/data` to reuse cached data.

### Directory Structure

```
/workspace/group/matbench/
├── data/                          ← SHARED dataset cache (reused across experiments)
├── experiments/                   ← ALL experiments live here
│   ├── 2026-03-23_cgcnn_baseline/
│   │   ├── config.json            ← experiment metadata (auto-generated)
│   │   ├── scripts/               ← training scripts used (frozen copy)
│   │   ├── models/                ← .pt checkpoints
│   │   ├── results/               ← matbench results .json.gz
│   │   ├── logs/                  ← training logs + TensorBoard
│   │   └── plots/                 ← visualizations
│   ├── 2026-03-24_mace_mlp_hpopt/
│   │   └── ...
│   └── 2026-03-25_alignn_v2/
│       └── ...
├── registry.json                  ← index of ALL experiments (append-only)
├── repos/                         ← cloned SOTA model repos (shared)
└── best/                          ← symlinks to best result per task (auto-updated)
```

### Experiment Naming Convention

Format: `YYYY-MM-DD_<model>_<description>`

Examples:
- `2026-03-23_cgcnn_baseline` — first CGCNN run
- `2026-03-24_cgcnn_lr1e3_hidden512` — CGCNN with specific hyperparameters
- `2026-03-25_mace_mlp_ensemble5seed` — MACE+MLP with 5-seed ensemble
- `2026-03-26_alignn_mp_e_form_only` — ALIGNN on a single task

### Experiment Setup Code (MANDATORY at start of every training script)

```python
import os, json
from datetime import datetime

# --- Experiment setup (COPY THIS BLOCK TO EVERY SCRIPT) ---
EXP_NAME = f"{datetime.now().strftime('%Y-%m-%d')}_{MODEL_NAME}_{DESCRIPTION}"
EXP_DIR = f"/workspace/group/matbench/experiments/{EXP_NAME}"
os.makedirs(f"{EXP_DIR}/models", exist_ok=True)
os.makedirs(f"{EXP_DIR}/results", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs", exist_ok=True)
os.makedirs(f"{EXP_DIR}/plots", exist_ok=True)
os.makedirs(f"{EXP_DIR}/scripts", exist_ok=True)

# Save experiment metadata
config = {
    "name": EXP_NAME,
    "created": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "tasks": TASKS,
    "hyperparameters": HYPERPARAMS,
    "gpu": "A100-SXM4-80GB",
    "status": "running"
}
with open(f"{EXP_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Append to global registry
REGISTRY = "/workspace/group/matbench/registry.json"
registry = json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {"experiments": []}
registry["experiments"].append({"name": EXP_NAME, "created": config["created"], "model": MODEL_NAME, "status": "running"})
with open(REGISTRY, "w") as f:
    json.dump(registry, f, indent=2)

# Copy this script to the experiment for reproducibility
import shutil
shutil.copy2(__file__, f"{EXP_DIR}/scripts/")

# Set paths — ALL outputs go to EXP_DIR
MODELS_DIR = f"{EXP_DIR}/models"
RESULTS_DIR = f"{EXP_DIR}/results"
LOGS_DIR = f"{EXP_DIR}/logs"
PLOTS_DIR = f"{EXP_DIR}/plots"
# --- End experiment setup ---

# Shared data cache (no duplication)
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.environ["MATMINER_DATA"] = "/workspace/group/matbench/data"
```

### After Training: Update Experiment Status

```python
# At end of training script
config["status"] = "completed"
config["completed"] = datetime.now().isoformat()
config["scores"] = {...}  # summary scores
with open(f"{EXP_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Update registry
for exp in registry["experiments"]:
    if exp["name"] == EXP_NAME:
        exp["status"] = "completed"
with open(REGISTRY, "w") as f:
    json.dump(registry, f, indent=2)
```

### Listing Past Experiments

```python
import json
registry = json.load(open("/workspace/group/matbench/registry.json"))
for exp in registry["experiments"]:
    print(f"{exp['name']}  [{exp['status']}]  {exp.get('model', '?')}")
```

## MatBench Tasks

| Task | Input | Output | Samples | Property |
|------|-------|--------|---------|----------|
| matbench_steels | Composition | Regression | 312 | Yield strength (MPa) |
| matbench_jdft2d | Structure | Regression | 636 | Exfoliation energy (meV/atom) |
| matbench_phonons | Structure | Regression | 1,265 | Max phonon freq (1/cm) |
| matbench_expt_gap | Composition | Regression | 4,604 | Band gap (eV) |
| matbench_dielectric | Composition | Regression | 4,764 | Dielectric constant |
| matbench_expt_is_metal | Composition | Classification | 4,921 | Metal / non-metal |
| matbench_glass | Composition | Classification | 5,680 | Glass-forming ability |
| matbench_log_gvrh | Structure | Regression | 10,987 | log10 bulk modulus (GPa) |
| matbench_log_kvrh | Structure | Regression | 10,987 | log10 shear modulus (GPa) |
| matbench_perovskites | Structure | Regression | 18,928 | Formation energy (eV) |
| matbench_mp_gap | Structure | Regression | 106,113 | Band gap (eV) |
| matbench_mp_is_metal | Structure | Classification | 106,113 | Metal / non-metal |
| matbench_mp_e_form | Structure | Regression | 132,752 | Formation energy (eV/atom) |

## Sub-Skills

| Sub-Skill | Directory | Description |
|-----------|-----------|-------------|
| task-setup | `task-setup/` | Load MatBench tasks, explore data, visualize distributions, understand 5-fold CV protocol |
| composition-models | `composition-models/` | Magpie featurization + RF/GBR/MLP for composition-input tasks |
| structure-gnn | `structure-gnn/` | Graph neural networks (CGCNN, SchNet, DimeNet++, ALIGNN) for structure-input tasks |
| sota-reproduction | `sota-reproduction/` | Reproduce published SOTA results (MODNet, coGN, MACE-MP-0, ALIGNN) |
| training-pipeline | `training-pipeline/` | Training loops, learning rate schedules, early stopping, TensorBoard logging |
| evaluation-submission | `evaluation-submission/` | Evaluate models, generate benchmark JSON, compare to leaderboard, prepare official submissions |
| sota-reference | `sota-reference/` | **READ BEFORE REPORTING RESULTS.** Authoritative SOTA scores for all 13 tasks with top-3 models, metrics, and repo links. Prevents incorrect SOTA comparisons. |
| model-optimization | `model-optimization/` | Hyperparameter search, ensembling, data augmentation, push beyond SOTA |

## Method Decision Guide

```
Start here: What is your task input type?
│
├─ Composition string?
│  └─ → composition-models/
│     (RF/GBR + Magpie features, or MLP on Magpie)
│
├─ Crystal structure?
│  └─ → structure-gnn/
│     (CGCNN, SchNet, DimeNet++, ALIGNN)
│
├─ Want to reproduce a published SOTA result?
│  └─ → sota-reproduction/
│     (MODNet, coGN, MACE-MP-0, ALIGNN configs)
│
├─ Need help with training loop / scheduler / logging?
│  └─ → training-pipeline/
│     (PyTorch training, TensorBoard, early stopping)
│
├─ Need correct SOTA numbers for comparison?
│  └─ → sota-reference/  ⚠️  ALWAYS READ THIS BEFORE REPORTING RESULTS
│     (Authoritative top-3 scores, metrics, model repos for all 13 tasks)
│
├─ Ready to evaluate and submit results?
│  └─ → evaluation-submission/
│     (Benchmark JSON, leaderboard comparison, submission)
│
└─ Want to push beyond current SOTA?
   └─ → model-optimization/
      (Hyperparameter search, ensembling, augmentation)
```

## Feishu Command Examples (飞书指令示例)

```
1. "帮我加载 matbench 全部 13 个任务，给我数据概览表"
2. "在 matbench_mp_e_form 上用 CGCNN 跑 5-fold benchmark，保存结果"
3. "参照 ALIGNN 配置，在 matbench_perovskites 上训练，目标超过 0.0269 MAE"
4. "对全部 8 个结构任务用 CGCNN+SchNet+DimeNet++ 集成，保存完整 benchmark"
5. "在 matbench_steels 上用随机森林+Magpie 特征跑 benchmark，分析特征重要性"
6. "加载之前的 benchmark 结果，和排行榜 SOTA 做对比图"
7. "搜索最新的 matbench 排行榜，找到当前 SOTA 模型，尝试复现"
8. "在 matbench_mp_e_form 上做超参优化，搜索最优学习率和模型配置"
```
