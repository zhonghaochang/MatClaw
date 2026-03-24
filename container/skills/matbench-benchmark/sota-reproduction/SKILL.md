# Reproduce SOTA Models on MatBench

## When to Use

- Reproduce current leaderboard SOTA results on MatBench tasks
- Optimize / fine-tune / ensemble existing SOTA models
- Clone a SOTA repo, set up its environment, and run matbench benchmarks
- Compare your model against verified SOTA baselines

## CRITICAL RULES

1. **Environment isolation**: If a SOTA model's dependencies conflict with the matbench env, **create a dedicated conda environment** for that model. NEVER downgrade packages in the matbench env.
2. **matbench in every env**: Every isolated conda environment MUST also install `matbench` and `matminer` — the matbench API (`task.load()`, `task.record()`, `mb.to_file()`) must run in the same Python process as the training code.
3. **CUDA torch**: Every new conda environment MUST install the CUDA version of PyTorch. NEVER install CPU-only torch.
4. **Shared data cache**: ALL environments share `/workspace/group/matbench/data/` for dataset cache. Set `MATBENCH_DATA_HOME` and `MATMINER_DATA` to this path. Data is already downloaded there — **NEVER delete or overwrite files in this directory**.
5. **Mirror fallback**: If pip/conda downloads are slow or fail, switch to a mirror (Tsinghua, USTC, Aliyun). See Section: Download Mirrors.
6. **Experiment directory**: All outputs go to `/workspace/group/matbench/experiments/YYYY-MM-DD_<model>_<description>/`. NEVER write to `/tmp/`.
7. **SOTA reference**: Before reporting results, read `~/.claude/skills/matbench-benchmark/sota-reference/SKILL.md` for correct SOTA numbers.

## Shared Data Cache (DO NOT TOUCH)

The following datasets are already cached at `/workspace/group/matbench/data/`. They are shared across ALL conda environments. **NEVER delete, move, or overwrite these files.**

```
/workspace/group/matbench/data/
├── matbench_dielectric.json.gz
├── matbench_jdft2d.json.gz
├── matbench_log_gvrh.json.gz
├── matbench_log_kvrh.json.gz
├── matbench_mp_e_form.json.gz
├── matbench_mp_gap.json.gz
├── matbench_mp_is_metal.json.gz
├── matbench_perovskites.json.gz
├── matbench_phonons.json.gz
├── matbench_steels.json.gz
├── matbench_expt_gap.json.gz
├── matbench_expt_is_metal.json.gz
├── matbench_glass.json.gz
├── graphs/           ← pre-built graph data (PyG format)
└── mace_features/    ← pre-extracted MACE embeddings
```

To use cached data, set these env vars **before importing matbench**:

```python
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.environ["MATMINER_DATA"] = "/workspace/group/matbench/data"

# Now import — matbench will find cached data, no re-download
from matbench.bench import MatbenchBenchmark
```

If a dataset is missing (e.g., you only ran a subset before), matbench will auto-download it to this directory. That's fine — it adds without overwriting.

## Method Decision Guide

```
What do you want to reproduce?

Model already installed in matbench env?
  chgnet, matgl, tpot, kgcnn → Use directly (Section 1)

Model needs isolated environment?
  ALIGNN (needs torch<=2.2 + dgl) → Create conda env (Section 2)
  MODNet (needs TensorFlow)       → Create conda env (Section 2)
  CrabNet (needs Python<3.11)     → Create conda env (Section 2)
  DeeperGATGNN (2021-era deps)    → Create conda env (Section 2)

Unknown model from a paper/repo?
  → Section 3: Full reproduction workflow
```

## Pre-installed SOTA Packages (matbench env)

These can be used directly with `/opt/conda/envs/matbench/bin/python`:

| Package | Version | Models | Status |
|---------|---------|--------|--------|
| chgnet | 0.4.2 | CHGNet universal potential | ✅ Ready |
| matgl | 2.1.1 | M3GNet, MEGNet (PyTorch) | ✅ Ready |
| tpot | 1.1.0 | TPOT-Mat AutoML | ✅ Ready |
| kgcnn | 4.0.2 | coGN, coNGN, MegNet, DimeNet++ | ✅ Ready |
| alignn | 2025.4.1 | ALIGNN (installed but dgl incompatible) | ⚠️ Needs isolated env |

## Section 1: Using Pre-installed Models

### CHGNet Example

```python
#!/opt/conda/envs/matbench/bin/python
"""CHGNet for MatBench structure tasks using pre-trained embeddings + MLP head."""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
os.environ["MATMINER_DATA"] = "/workspace/group/matbench/data"

import json, numpy as np, torch
from datetime import datetime
from matbench.bench import MatbenchBenchmark
from chgnet.model import CHGNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Experiment setup ---
MODEL_NAME = "chgnet"
DESCRIPTION = "pretrained_embeddings"
TASK_NAME = "matbench_mp_e_form"
EXP_NAME = f"{datetime.now().strftime('%Y-%m-%d')}_{MODEL_NAME}_{DESCRIPTION}"
EXP_DIR = f"/workspace/group/matbench/experiments/{EXP_NAME}"
for d in ["models", "results", "logs", "plots", "scripts"]:
    os.makedirs(f"{EXP_DIR}/{d}", exist_ok=True)

# Load pre-trained CHGNet
chgnet = CHGNet.load()
print(f"CHGNet loaded: {sum(p.numel() for p in chgnet.parameters())} params")

# ... (training loop using matbench 5-fold protocol)
# Save results to EXP_DIR/results/
```

### MatGL (M3GNet) Example

```python
#!/opt/conda/envs/matbench/bin/python
"""M3GNet via matgl for MatBench."""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

import matgl
import torch
from matgl.utils.training import ModelLightningModule
import lightning as L

# Load pre-trained M3GNet
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
print(f"M3GNet loaded")

# ... (adapt to matbench protocol)
```

## Section 2: Creating Isolated Environments for Incompatible Models

**This is the core procedure when a SOTA model's dependencies conflict with the matbench env.**

### Step-by-step Protocol

```bash
#!/bin/bash
# === Isolated Environment Creation Protocol ===
# Run this INSIDE the container

MODEL_NAME="alignn"           # Change per model
PY_VERSION="3.10"             # Check model repo for Python version
TORCH_VERSION="2.2.1"         # Check model repo for torch version
CUDA_TAG="cu121"              # Match torch version's CUDA support

ENV_NAME="${MODEL_NAME}_env"
ENV_BIN="/opt/conda/envs/${ENV_NAME}/bin"

echo "=== Creating isolated env: ${ENV_NAME} ==="

# Step 1: Create conda environment
conda create -n ${ENV_NAME} python=${PY_VERSION} -y -c conda-forge --override-channels

# Step 2: Install CUDA PyTorch (MANDATORY — never CPU-only)
${ENV_BIN}/pip install torch==${TORCH_VERSION} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG}

# Step 3: Verify CUDA
${ENV_BIN}/python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Step 4: Install the model package and its dependencies
${ENV_BIN}/pip install ${MODEL_NAME}

# Step 5: Install matbench + matminer in this env (MANDATORY)
# The matbench API (task.load, task.record, mb.to_file) MUST run
# in the same Python process as the training code.
# Without this, you cannot load data or record predictions.
${ENV_BIN}/pip install matbench matminer

# Step 5b: Point data cache to shared directory (no re-download)
echo "Set MATBENCH_DATA_HOME=/workspace/group/matbench/data in your scripts"

# Step 6: Verify
${ENV_BIN}/python -c "
import ${MODEL_NAME}; print(f'${MODEL_NAME} OK')
import matbench; print(f'matbench OK')
import torch; print(f'CUDA: {torch.cuda.is_available()}')
"

echo "=== Environment ${ENV_NAME} ready ==="
echo "Usage: ${ENV_BIN}/python your_script.py"
```

### Model-Specific Environment Recipes

#### ALIGNN (needs torch<=2.2.1 + dgl)

```bash
conda create -n alignn_env python=3.10 -y -c conda-forge --override-channels
/opt/conda/envs/alignn_env/bin/pip install torch==2.2.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
/opt/conda/envs/alignn_env/bin/pip install dgl==1.1.3 \
    -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
/opt/conda/envs/alignn_env/bin/pip install alignn matbench matminer
# Verify
/opt/conda/envs/alignn_env/bin/python -c "
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import torch; print(f'ALIGNN OK, CUDA: {torch.cuda.is_available()}')
"
```

#### MODNet (needs TensorFlow)

```bash
conda create -n modnet_env python=3.10 -y -c conda-forge --override-channels
/opt/conda/envs/modnet_env/bin/pip install tensorflow==2.15
/opt/conda/envs/modnet_env/bin/pip install modnet matbench matminer
# Verify
/opt/conda/envs/modnet_env/bin/python -c "
from modnet.models import MODNetModel; print('MODNet OK')
"
```

#### CrabNet (needs Python<3.11)

```bash
conda create -n crabnet_env python=3.9 -y -c conda-forge --override-channels
/opt/conda/envs/crabnet_env/bin/pip install torch==2.2.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu121
/opt/conda/envs/crabnet_env/bin/pip install crabnet matbench matminer
# Verify
/opt/conda/envs/crabnet_env/bin/python -c "
from crabnet.crabnet_ import CrabNet; print('CrabNet OK')
import torch; print(f'CUDA: {torch.cuda.is_available()}')
"
```

#### DeeperGATGNN (git clone, 2021-era deps)

```bash
conda create -n gatgnn_env python=3.9 -y -c conda-forge --override-channels
/opt/conda/envs/gatgnn_env/bin/pip install torch==1.13.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu117
/opt/conda/envs/gatgnn_env/bin/pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
cd /workspace/group/matbench/repos
git clone https://github.com/usccolumbia/deeperGATGNN.git
/opt/conda/envs/gatgnn_env/bin/pip install matbench matminer
# Verify
/opt/conda/envs/gatgnn_env/bin/python -c "
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"
```

### IMPORTANT: Environment Usage in Training Scripts

When using an isolated env, the script shebang and all subprocess calls must use that env's Python:

```python
#!/opt/conda/envs/alignn_env/bin/python    # <-- NOT matbench env
"""ALIGNN training on matbench."""
import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
# ... rest of training code
```

### IMPORTANT: Data and Results Stay on Data Disk

Even with a different conda env, file paths remain the same:

```python
# Experiment dir (data disk, persistent)
EXP_DIR = "/workspace/group/matbench/experiments/2026-03-25_alignn_mp_e_form/"

# Shared data cache (reused across all envs)
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
```

## Section 3: Full Reproduction Workflow (Unknown Model)

When you encounter a model you haven't seen before:

### Step 1: Research

```
WebSearch: "{model_name} matbench benchmark github"
WebSearch: "{model_name} crystal property prediction code"
WebFetch: https://github.com/{org}/{repo}  (read README)
```

### Step 2: Clone and Inspect

```bash
cd /workspace/group/matbench/repos/
git clone --depth 1 {repo_url} {model_name}

# Find matbench-related code
find {model_name}/ -name "*.py" | xargs grep -l "matbench\|MatbenchBenchmark" 2>/dev/null
cat {model_name}/requirements.txt 2>/dev/null || cat {model_name}/setup.py 2>/dev/null
```

### Step 3: Compatibility Check

```python
#!/opt/conda/envs/matbench/bin/python
"""Check if repo deps are compatible with matbench env."""
import subprocess, re

# Read requirements
with open("/workspace/group/matbench/repos/{model_name}/requirements.txt") as f:
    reqs = f.read()

print("=== Repo Requirements ===")
print(reqs)

# Check each against installed
for line in reqs.strip().split("\n"):
    pkg = re.split(r'[><=!~]', line.strip())[0].strip()
    if not pkg or pkg.startswith("#"):
        continue
    result = subprocess.run(
        ["/opt/conda/envs/matbench/bin/pip", "show", pkg],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ver = [l for l in result.stdout.split("\n") if l.startswith("Version:")][0]
        print(f"  ✅ {pkg}: {ver}  (req: {line.strip()})")
    else:
        print(f"  ❌ {pkg}: NOT INSTALLED  (req: {line.strip()})")
```

### Step 4: Decide Environment Strategy

```
All deps compatible with matbench env?
  YES → pip install directly in matbench env
  NO  → Create isolated conda env (Section 2 protocol)

Specific conflicts:
  torch version mismatch    → isolated env with correct torch+CUDA
  TensorFlow required       → isolated env (TF + torch can't coexist cleanly)
  Python version mismatch   → isolated env with correct Python
  Only minor version diffs  → try --no-deps install first, test if it works
```

### Step 5: Run Benchmark

```python
# Use the correct env's Python!
# /opt/conda/envs/{model_name}_env/bin/python  OR  /opt/conda/envs/matbench/bin/python

import os
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"
from datetime import datetime

MODEL_NAME = "{model_name}"
TASK_NAME = "{task_name}"
EXP_NAME = f"{datetime.now().strftime('%Y-%m-%d')}_{MODEL_NAME}_{TASK_NAME}"
EXP_DIR = f"/workspace/group/matbench/experiments/{EXP_NAME}"
for d in ["models", "results", "logs", "plots", "scripts"]:
    os.makedirs(f"{EXP_DIR}/{d}", exist_ok=True)

# ... standard matbench 5-fold protocol ...
# Save to EXP_DIR/results/
```

## Download Mirrors (When Network is Slow)

If pip downloads are slow (< 100 KB/s) or timeout, switch mirrors:

```bash
# Tsinghua mirror (recommended for China)
pip install {package} -i https://pypi.tuna.tsinghua.edu.cn/simple

# USTC mirror
pip install {package} -i https://pypi.mirrors.ustc.edu.cn/simple

# Aliyun mirror
pip install {package} -i https://mirrors.aliyun.com/pypi/simple

# PyTorch with Tsinghua mirror
pip install torch==2.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Conda with Tsinghua mirror
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
```

**Detection rule**: If any single download takes > 60 seconds or speed drops below 100 KB/s, switch to a mirror immediately. Don't wait for timeout.

## Reproduction Checklist

Before starting, verify:
- [ ] Identified the model's repo and exact version used on leaderboard
- [ ] Checked dependency compatibility with matbench env
- [ ] Created isolated conda env if needed (with CUDA torch!)
- [ ] Created experiment directory under `experiments/`
- [ ] Verified CUDA is available in the chosen env
- [ ] Estimated training time on A100 (check paper/README)
- [ ] Read `sota-reference/SKILL.md` for correct SOTA target

After completion, verify:
- [ ] Results saved to `experiments/{name}/results/`
- [ ] Model checkpoints saved to `experiments/{name}/models/`
- [ ] Training script copied to `experiments/{name}/scripts/`
- [ ] config.json created with hyperparameters and scores
- [ ] registry.json updated

## Common Issues

| Issue | Cause | Solution |
|-------|-------|---------|
| `pip install` takes forever | Network throttled or blocked | Switch to Tsinghua/USTC/Aliyun mirror |
| `dgl` import fails with graphbolt error | dgl version doesn't match torch ABI | Create isolated env with matching torch+dgl versions |
| CUDA out of memory | Batch too large | Reduce batch_size, use gradient accumulation, use `torch.cuda.amp` |
| `No module named 'pkg_resources'` | setuptools >= 72 removed it | `pip install "setuptools<71"` |
| conda create fails with ToS error | Anaconda ToS not accepted | Add `-c conda-forge --override-channels` |
| torch is CPU-only after install | Installed from wrong index | Reinstall with `--index-url https://download.pytorch.org/whl/cu121` |
| Isolated env takes too long to create | Large packages (torch ~2GB) | Use pip mirror; accept ~10-15 min setup time |
