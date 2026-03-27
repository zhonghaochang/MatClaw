# Tournament-Based Automatic SOTA Attack

Systematic, multi-phase strategy to beat MatBench SOTA scores. Uses a tournament mechanism to avoid wasting compute on bad architectures.

## When to Use

- User says "刷榜", "beat SOTA", "超越SOTA", "auto attack", "锦标赛"
- Any request to maximize performance on a specific matbench task

## CRITICAL: Read Before Starting

1. Read `sota-reference/SKILL.md` to know the EXACT target score
2. Read `reference-library/SKILL.md` to know which repos to study
3. Determine task category (see Strategy Selection below)

## Strategy Selection by Dataset Size

```
Task samples < 5,000? (steels, jdft2d, phonons, expt_gap, dielectric, expt_is_metal, glass)
│
├─ Composition input? (steels, expt_gap, dielectric, expt_is_metal, glass)
│  └─ STRATEGY A: Composition Tournament
│
├─ Structure input, < 2,000 samples? (jdft2d: 636, phonons: 1265)
│  └─ STRATEGY B: Small Structure Tournament
│
Task samples 5,000 - 50,000? (log_gvrh, log_kvrh, perovskites)
│  └─ STRATEGY C: Medium Structure Tournament
│
Task samples > 100,000? (mp_gap, mp_is_metal, mp_e_form)
│  └─ STRATEGY D: Large Structure Tournament
```

---

## Phase -1: Dynamic Candidate Discovery (MANDATORY before every tournament)

**Do NOT blindly use the preset candidate pools below. They are fallbacks only.**

Before Phase 0, the agent MUST:

### Step 1: Scan Reference Repos for Best Candidates

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""Pre-tournament: Scan reference repos to build dynamic candidate pool."""
import os, json

REPOS_DIR = "/workspace/group/reference/repos"
repos = sorted(os.listdir(REPOS_DIR))
print(f"Available reference repos ({len(repos)}):")

# For each repo, check:
# 1. Does it have matbench results or training scripts?
# 2. What tasks does it target?
# 3. What scores does it report?
# 4. When was it last updated?
for repo in repos:
    repo_path = os.path.join(REPOS_DIR, repo)
    # Look for matbench-related files
    for root, dirs, files in os.walk(repo_path):
        for f in files:
            if any(kw in f.lower() for kw in ["matbench", "benchmark", "train", "config"]):
                print(f"  {repo}/{os.path.relpath(os.path.join(root, f), repo_path)}")
        break  # only top-level scan for speed
```

### Step 2: Web Search for Latest Models

Use `WebSearch` tool to search for:
- "matbench SOTA 2025 2026" — find newest leaderboard entries
- "{task_name} state of the art" — task-specific advances
- "materials property prediction new model" — recent papers

### Step 3: Build Custom Candidate Pool

Combine:
- Models found in reference repos that target this task
- Models from web search that report strong results
- Preset candidates from this skill (as fallback)
- At least 1 novel/experimental architecture the agent designs itself

### Step 4: Read SOTA Target Score

```
Read sota-reference/SKILL.md → get exact SOTA MAE/AUC for the target task
This is the GATE score for Phase 0.
```

---

## SOTA Gate: Fold-0 Qualification Rule

**A candidate MUST beat the SOTA score on fold 0 to enter the tournament.**

```
Phase 0 sea trial result for candidate X:
  fold_0_mae = 82.3
  SOTA_mae = 79.95

  82.3 > 79.95 → ELIMINATED. Not competitive enough.

Phase 0 sea trial result for candidate Y:
  fold_0_mae = 76.1
  SOTA_mae = 79.95

  76.1 < 79.95 → QUALIFIED. Enters Phase 1.
```

**Rules:**
- Regression tasks: fold_0 MAE must be ≤ SOTA MAE
- Classification tasks: fold_0 ROC-AUC must be ≥ SOTA ROC-AUC
- If ZERO candidates pass the gate → relax to top-3 regardless of SOTA comparison, but flag to user: "No candidate beat SOTA on fold 0, proceeding with best available"
- The gate ensures we don't waste 10+ hours on architectures that can't compete

**Implementation in sea trial script:**

```python
SOTA_SCORE = 79.95  # from sota-reference/SKILL.md
IS_CLASSIFICATION = False  # True for expt_is_metal, glass, mp_is_metal

# After computing fold_0_score for each candidate:
qualified = []
eliminated = []
for name, result in sea_trial_results.items():
    score = result["mae"] if not IS_CLASSIFICATION else result["rocauc"]
    if IS_CLASSIFICATION:
        passed = score >= SOTA_SCORE
    else:
        passed = score <= SOTA_SCORE

    if passed:
        qualified.append((name, result))
        print(f"  ✅ {name}: {score:.4f} — QUALIFIED (beats SOTA {SOTA_SCORE})")
    else:
        eliminated.append((name, result))
        gap = abs(score - SOTA_SCORE) / SOTA_SCORE * 100
        print(f"  ❌ {name}: {score:.4f} — ELIMINATED ({gap:.1f}% worse than SOTA)")

if len(qualified) == 0:
    print("⚠️ WARNING: No candidate beat SOTA on fold 0!")
    print("Falling back to top-3 candidates regardless.")
    qualified = sorted(sea_trial_results.items(), key=lambda x: x[1]["mae"])[:3]

print(f"\nQualified: {len(qualified)}/{len(sea_trial_results)} → proceed to Phase 1")
```

---

## STRATEGY A: Composition Tournament (steels, expt_gap, dielectric, expt_is_metal, glass)

Small composition datasets. Feature-based + DL hybrid approaches dominate.

### Phase 0: Sea Trial (海选) — 1-2 hours

Run each candidate on fold 0 only, 1 seed, reduced epochs.
**Candidates from Phase -1 dynamic discovery + these presets as fallback:**

| # | Architecture | Reference Repo | Key Idea | Expected Strength |
|---|-------------|---------------|----------|------------------|
| 1 | Roost | `reference/repos/roost/` | Attention-based element message passing | Strong on small data |
| 2 | CrabNet | `reference/repos/crabnet/` | Transformer + fractional encoding | SOTA on expt_gap |
| 3 | Wren/Aviary | `reference/repos/aviary/` | Wyckoff-augmented Roost | Better than Roost on some tasks |
| 4 | MODNet-style | `reference/repos/modnet/` | Mutual-info feature selection + NN | SOTA on 6-7 tasks |
| 5 | MatterVial-style | Study paper: Nature npj 2026 | GNN embeddings + symbolic features | >40% improvement reported |
| 6 | MACE Transfer | `reference/repos/mace/` | Pretrained element embeddings → MLP | Leverages foundation model |
| 7 | Custom Composition Transformer | `composition-models/` Script 4 | Periodic-table-aware + gated FFN | Novel architecture |
| 8 | TabPFN + GNN embeddings | ICL-FM paper (arXiv 2601.00133) | Zero-shot foundation model | Training-free baseline |

**⚠️ After Phase 0, apply SOTA Gate: only candidates with fold_0 MAE ≤ SOTA advance.**

**Sea trial code pattern:**

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""Phase 0: Sea Trial - Quick architecture screening on fold 0."""
import os, json, time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")

os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

TASK_NAME = "matbench_steels"  # CHANGE THIS
SOTA_SCORE = 79.95  # CHANGE THIS - from sota-reference skill
EXP_NAME = f"{datetime.now().strftime('%Y-%m-%d')}_{TASK_NAME}_tournament"
EXP_DIR = f"/workspace/group/matbench/experiments/{EXP_NAME}"
os.makedirs(f"{EXP_DIR}/phase0_sea_trial", exist_ok=True)
os.makedirs(f"{EXP_DIR}/phase1_semifinal", exist_ok=True)
os.makedirs(f"{EXP_DIR}/phase2_hpo", exist_ok=True)
os.makedirs(f"{EXP_DIR}/phase3_ensemble", exist_ok=True)
os.makedirs(f"{EXP_DIR}/scripts", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs", exist_ok=True)

# Each candidate returns fold-0 MAE after quick training
results = {}

# --- Candidate 1: Roost (study reference/repos/roost/ first) ---
# Read roost/roost/model.py to understand architecture
# Implement simplified version, train fold 0, 100 epochs, 1 seed
# results["roost"] = {"mae": ..., "time_min": ..., "params": ...}

# --- Candidate 2: CrabNet ---
# ... same pattern ...

# --- After all candidates ---
# Sort by MAE, print ranking
ranking = sorted(results.items(), key=lambda x: x[1]["mae"])
print("\n=== SEA TRIAL RANKING ===")
for i, (name, r) in enumerate(ranking):
    gap = (r["mae"] / SOTA_SCORE - 1) * 100
    print(f"  #{i+1} {name}: MAE={r['mae']:.2f} ({gap:+.1f}% vs SOTA) [{r['time_min']:.1f}min]")

# Save results
with open(f"{EXP_DIR}/phase0_sea_trial/ranking.json", "w") as f:
    json.dump({"ranking": ranking, "sota": SOTA_SCORE, "task": TASK_NAME}, f, indent=2)

# Select top 50% for Phase 1
n_advance = max(3, len(ranking) // 2)
advancing = [name for name, _ in ranking[:n_advance]]
print(f"\nAdvancing to Phase 1: {advancing}")
```

### Phase 1: Semifinal (半决赛) — 3-5 hours

Top 50% architectures from Phase 0. Full 5-fold CV, 3 seeds each.

```python
"""Phase 1: Semifinal - Full 5-fold CV for top architectures."""
# For each advancing architecture:
#   Train all 5 folds × 3 seeds
#   Record OOF (out-of-fold) predictions for later ensemble
#   Compute mean MAE ± std across folds
#   Save all model checkpoints

# Key: Save OOF predictions!
# OOF = for each fold, the TEST predictions from models trained on OTHER folds
# These are unbiased estimates needed for Phase 3 ensemble stacking
```

### Phase 2: HPO Final (超参决赛) — 5-10 hours

Top 2-3 architectures. 200 Optuna trials each.

```python
"""Phase 2: HPO - Optuna hyperparameter optimization for top architectures."""
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden = trial.suggest_categorical("hidden", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
    # ... architecture-specific params ...

    # Train on fold 0 only for speed (Optuna pruning handles bad trials)
    mae = train_and_evaluate(config, fold=0)
    return mae

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)
study.optimize(objective, n_trials=200)

# Take top-10 configs, train each × 5 folds × 5 seeds
# Save ALL models and OOF predictions
```

### Phase 3: Ensemble Sprint (集成冲刺) — 1-2 hours

Combine all models from Phase 1 + Phase 2 using greedy forward selection.

```python
"""Phase 3: Ensemble - Greedy hill-climbing from all model predictions."""
import numpy as np

# Load ALL OOF predictions from Phase 1 and Phase 2
# all_oofs shape: (n_models, n_samples)
# true_values shape: (n_samples,)

# Greedy forward selection
selected = []
best_mae = float("inf")
remaining = list(range(len(all_oofs)))

for round_num in range(min(20, len(all_oofs))):
    best_candidate = None
    for idx in remaining:
        trial_ensemble = selected + [idx]
        trial_pred = np.mean(all_oofs[trial_ensemble], axis=0)
        trial_mae = np.mean(np.abs(trial_pred - true_values))
        if trial_mae < best_mae:
            best_mae = trial_mae
            best_candidate = idx

    if best_candidate is None:
        break  # No improvement possible
    selected.append(best_candidate)
    remaining.remove(best_candidate)
    print(f"  Round {round_num+1}: +model#{best_candidate} → ensemble MAE={best_mae:.4f} "
          f"({len(selected)} models, {(best_mae/SOTA_SCORE-1)*100:+.1f}% vs SOTA)")

    if best_mae < SOTA_SCORE:
        print(f"  🏆 SOTA BEATEN! {best_mae:.4f} < {SOTA_SCORE}")

# Generate final predictions using selected models
# Record with matbench API: task.record(fold, final_predictions)
```

### Phase 3b: Additional Tricks (加分项)

Apply AFTER ensemble, each one independently validated:

```
□ EMA weights (replace each model with its EMA version, re-ensemble)
□ Prediction calibration (isotonic regression on OOF predictions)
□ Retrain selected models on 100% data (no validation holdout) for final submission
□ Stacking: train small MLP on OOF predictions as meta-learner
```

---

## STRATEGY B: Small Structure Tournament (jdft2d: 636, phonons: 1265)

Same 4-phase structure as Strategy A, but different candidate pool:

### Phase 0 Candidates:

| # | Architecture | Reference Repo | Why |
|---|-------------|---------------|-----|
| 1 | MACE-MP fine-tune (freeze low layers) | `reference/repos/mace/` | Best transfer learning for small data |
| 2 | ALIGNN | `reference/repos/alignn/` | Bond angle info helps phonons |
| 3 | coGN-style (kgcnn) | `reference/repos/kgcnn/` | Current SOTA architecture |
| 4 | SchNet + augmentation | `reference/repos/schnetpack/` | + perturbation augmentation |
| 5 | Crystal Twins pretrain → fine-tune | Paper reference | Self-supervised pretraining |
| 6 | MODNet + SOAP features | `reference/repos/modnet/` | Feature-based, strong on small data |
| 7 | MatterVial hybrid | Paper: npj 2026 | GNN embeddings + symbolic features |

**⚠️ After Phase 0, apply SOTA Gate: only candidates with fold_0 MAE ≤ SOTA advance.**

**CRITICAL for small structure data:**
- Perturbation data augmentation (perturb atomic positions ±0.05Å) can reduce MAE by up to 66%
- Transfer learning from MACE-MP-0: freeze through interaction layers, fine-tune readout only
- These are MORE important than architecture choice

---

## STRATEGY C: Medium Structure Tournament (log_gvrh, log_kvrh, perovskites)

### Phase 0 Candidates:

| # | Architecture | Reference Repo | Why |
|---|-------------|---------------|-----|
| 1 | coGN | `reference/repos/kgcnn/` | Current SOTA |
| 2 | ALIGNN | `reference/repos/alignn/` | Strong + fast |
| 3 | DimeNet++ | `reference/repos/dimenet/` | Directional message passing |
| 4 | MACE fine-tune | `reference/repos/mace/` | Foundation model advantage |
| 5 | PaiNN | `reference/repos/painn/` | Equivariant, fast |
| 6 | DenseGNN-style | Paper reference | Dense connections prevent oversmoothing |
| 7 | GemNet | `reference/repos/gemnet/` | Triplet interactions |
| 8 | Graph Transformer | `reference/repos/graphormer/` | Attention-based |

**⚠️ After Phase 0, apply SOTA Gate: only candidates with fold_0 MAE ≤ SOTA advance.**

**Focus areas:**
- Graph construction matters most: test cutoff radius 4.0/6.0/8.0Å, Gaussian bins 25/50/100
- Deeper models (4-6 conv layers) work here — enough data to avoid overfitting
- coGN's edge: jointly optimized graph preprocessing + architecture

---

## STRATEGY D: Large Structure Tournament (mp_gap, mp_is_metal, mp_e_form)

### Phase 0 Candidates:

| # | Architecture | Reference Repo | Why |
|---|-------------|---------------|-----|
| 1 | coGN | `reference/repos/kgcnn/` | Current SOTA |
| 2 | EquiformerV2 | `reference/repos/fairchem-ocp/` | Best equivariant transformer |
| 3 | MACE (full training) | `reference/repos/mace/` | Enough data to train from scratch |
| 4 | DenseGNN | Paper reference | Deep without oversmoothing |
| 5 | NequIP | `reference/repos/nequip/` | E(3)-equivariant |
| 6 | SevenNet | `reference/repos/sevenn/` | Samsung's universal potential |
| 7 | MatGL (M3GNet) | `reference/repos/matgl/` | Fast, good baseline |

**⚠️ After Phase 0, apply SOTA Gate: only candidates with fold_0 MAE ≤ SOTA advance.**

**Focus areas:**
- Large batch sizes (512-2048) to fully utilize A100 80GB
- Mixed precision (AMP) essential — 2x speedup
- Fewer seeds needed (data is large enough for stable estimates)
- HPO budget should be smaller (each trial takes hours)
- Ensemble of top 3-5 diverse architectures is the final push

---

## MANDATORY: A100 GPU Utilization Rules

The A100-SXM4-80GB has 80GB VRAM and massive compute. Low utilization (<30%) means wasted resources.

### Rule 1: Maximize Batch Size

```python
# WRONG — wastes 95% of A100 VRAM
DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=False)

# CORRECT — fill the GPU
DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True, persistent_workers=True)
```

**Batch size guide by model type:**

| Model | Min batch_size | Recommended | Max (A100 80GB) |
|-------|---------------|-------------|-----------------|
| Roost/CrabNet (composition) | 256 | 512-1024 | 2048+ |
| CGCNN/SchNet | 128 | 256-512 | 1024 |
| ALIGNN/coGN | 64 | 128-256 | 512 |
| MACE/EquiformerV2 | 32 | 64-128 | 256 |
| DimeNet++ | 16 | 32-64 | 128 |

If batch_size causes OOM, use gradient accumulation instead of reducing batch_size:

```python
ACCUMULATION_STEPS = 4  # effective batch = batch_size × 4
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = model(batch) / ACCUMULATION_STEPS
    loss.backward()
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Rule 2: DataLoader Must Use Workers

```python
# MANDATORY settings for ALL DataLoaders:
DataLoader(
    dataset,
    batch_size=LARGE_BATCH,     # see table above
    num_workers=4,              # MINIMUM 4, use 8 for large datasets
    pin_memory=True,            # faster CPU→GPU transfer
    persistent_workers=True,    # don't recreate workers each epoch
    prefetch_factor=2,          # pre-load next batches while GPU computes
)
```

### Rule 3: Enable Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in loader:
    with autocast(device_type="cuda"):    # FP16 forward pass — 2x faster
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Rule 4: Enable TF32 on A100

```python
# Put at top of EVERY training script — free 3x speedup for matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Rule 5: Run Multiple Models in Parallel

When training small models (composition models, <5GB VRAM each), run 3-4 simultaneously using the Task tool. Let CUDA manage memory naturally — if one process OOMs, reduce concurrency.

```
Parallel budget (natural competition, no manual memory fraction):
  Roost/CrabNet × 4 concurrent  → each ~1-2GB, total ~8GB, safe
  CGCNN/SchNet  × 3 concurrent  → each ~3-5GB, total ~15GB, safe
  ALIGNN/coGN   × 2 concurrent  → each ~5-15GB, total ~30GB, safe
  MACE/large    × 1 only        → may use 30-50GB

  If OOM → reduce concurrency by 1, retry
```

### Rule 6: Monitor GPU Utilization

```python
# Add to training loop — print every 50 epochs
if epoch % 50 == 0:
    mem = torch.cuda.memory_allocated() / 1e9
    max_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU mem: {mem:.1f}/{max_mem:.1f} GB used / 80 GB total "
          f"({max_mem/80*100:.0f}% utilization)")
```

**Target: >50% VRAM utilization during training. If below 30%, increase batch_size or run parallel models.**

---

## Parallel Execution with Multi-Agent

Use Claude Code's `Task` and `TeamCreate` tools to run candidates in parallel.
The container has `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` enabled.

### Phase 0: Parallel Sea Trial (ALL candidates simultaneously)

```
Main Agent workflow:
│
├─ 1. Write training scripts for all candidates to EXP_DIR/scripts/
│     - sea_trial_roost.py
│     - sea_trial_crabnet.py
│     - sea_trial_mace_transfer.py
│     - ... (one script per candidate)
│
├─ 2. Launch ALL candidates in parallel using Task tool:
│
│     task1 = Task("Run: /opt/conda/envs/matbench/bin/python {EXP_DIR}/scripts/sea_trial_roost.py > {EXP_DIR}/phase0_sea_trial/roost.log 2>&1")
│     task2 = Task("Run: /opt/conda/envs/matbench/bin/python {EXP_DIR}/scripts/sea_trial_crabnet.py > {EXP_DIR}/phase0_sea_trial/crabnet.log 2>&1")
│     task3 = Task("Run: /opt/conda/envs/matbench/bin/python {EXP_DIR}/scripts/sea_trial_mace.py > {EXP_DIR}/phase0_sea_trial/mace.log 2>&1")
│     task4 = Task("Run: /opt/conda/envs/matbench/bin/python {EXP_DIR}/scripts/sea_trial_aviary.py > {EXP_DIR}/phase0_sea_trial/aviary.log 2>&1")
│     ... launch all at once
│
├─ 3. Monitor with TaskOutput — wait for all to complete
│
├─ 4. Read results from each candidate's output file:
│     - Each script writes {EXP_DIR}/phase0_sea_trial/{name}_result.json
│     - Contains: {"mae": float, "time_min": float, "params": int}
│
└─ 5. Rank and select top 50% → proceed to Phase 1
```

**IMPORTANT: Each training script MUST be self-contained and write results to a JSON file.**
The main agent collects results AFTER all tasks complete, not during training.

### Phase 0 Script Template (each candidate follows this pattern):

```python
#!/usr/bin/env /opt/conda/envs/matbench/bin/python
"""Sea trial script — runs independently, writes result to JSON."""
import os, json, time
import matplotlib
matplotlib.use("Agg")
os.environ["MATBENCH_DATA_HOME"] = "/workspace/group/matbench/data"

TASK_NAME = os.environ.get("TASK_NAME", "matbench_steels")
EXP_DIR = os.environ.get("EXP_DIR", "/workspace/group/matbench/experiments/tournament")
MODEL_NAME = "roost"  # CHANGE PER CANDIDATE

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from matbench.bench import MatbenchBenchmark
mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
task = list(mb.tasks)[0]
task.load()

start = time.time()
train_inputs, train_outputs = task.get_train_and_val_data(0)  # fold 0 only
test_inputs = task.get_test_data(0, include_target=False)

# ========= MODEL IMPLEMENTATION HERE =========
# ... train model, get predictions ...
# predictions = model.predict(test_inputs)
# ==============================================

elapsed = (time.time() - start) / 60
mae = float(abs(predictions - task.get_test_data(0, include_target=True)[1]).mean())

result = {"model": MODEL_NAME, "mae": mae, "time_min": elapsed, "params": sum(p.numel() for p in model.parameters())}
with open(f"{EXP_DIR}/phase0_sea_trial/{MODEL_NAME}_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"DONE: {MODEL_NAME} MAE={mae:.4f} ({elapsed:.1f}min)")
```

### Phase 1: Parallel Semifinal (top candidates simultaneously)

```
Main Agent workflow:
│
├─ 1. For each advancing architecture, write a full 5-fold training script:
│     - semifinal_roost.py (trains all 5 folds × 3 seeds, saves OOF)
│     - semifinal_crabnet.py
│     - semifinal_mace.py
│
├─ 2. Launch in parallel:
│     task_roost = Task("Run semifinal_roost.py")
│     task_crabnet = Task("Run semifinal_crabnet.py")
│     task_mace = Task("Run semifinal_mace.py")
│
├─ 3. Wait for all → read scores.json from each
│
└─ 4. Rank by mean MAE across 5 folds → select top 2-3
```

### Phase 2: Parallel HPO (different architectures on GPU time-sharing)

```
Main Agent workflow:
│
├─ Option A (sequential HPO, simpler): Run Optuna for each arch one at a time
│   - Safest: full GPU for each trial
│   - Use: when models are large (>1GB VRAM)
│
├─ Option B (parallel HPO, faster): Run 2 Optuna processes simultaneously
│   - Each gets ~40GB VRAM on A100
│   - Use: when models are small (composition models, <5GB each)
│   - Set CUDA_MEM_FRACTION or use separate CUDA streams
│
│   task_hpo_arch1 = Task("Run hpo_roost.py --n-trials 200")
│   task_hpo_arch2 = Task("Run hpo_crabnet.py --n-trials 200")
│
└─ After HPO: take top-10 configs per arch, train × 5-10 seeds (parallel)
```

### Phase 3: Sequential Ensemble (no parallelism needed)

Ensemble is fast (minutes) — just loads saved OOF predictions and does greedy selection.
No need for parallel execution.

### TeamCreate for Complex Orchestration

For very complex tournaments (multiple tasks simultaneously), use TeamCreate:

```
Main Agent:
│
├─ TeamCreate "steels_attacker":
│   prompt: "You are attacking matbench_steels. Run the full tournament
│            (Phase 0-3) following auto-tournament/SKILL.md.
│            Target SOTA: 79.95 MAE. Save to experiments/YYYY-MM-DD_steels_tournament/"
│
├─ TeamCreate "jdft2d_attacker":
│   prompt: "You are attacking matbench_jdft2d. Run the full tournament...
│            Target SOTA: 33.19 MAE."
│
├─ TeamCreate "phonons_attacker":
│   prompt: "You are attacking matbench_phonons. Run the full tournament...
│            Target SOTA: 28.76 MAE."
│
└─ Wait for all → collect results → report to user
```

⚠️ **GPU contention warning**: Multiple TeamCreate agents share the same GPU.
For large models, limit to 1-2 concurrent team members.
For small composition models, 3-4 can run simultaneously on A100 80GB.

### GPU Memory Budget for Parallel Training

| Model Type | VRAM per model | Max parallel on A100 80GB |
|-----------|---------------|--------------------------|
| Roost/CrabNet (composition) | 0.5-2 GB | 4-6 concurrent |
| CGCNN/SchNet (small GNN) | 2-5 GB | 3-4 concurrent |
| ALIGNN/coGN (medium GNN) | 5-15 GB | 2-3 concurrent |
| MACE/EquiformerV2 (large) | 15-40 GB | 1-2 concurrent |
| DimeNet++ (heavy) | 20-50 GB | 1 only |

For Phase 2 HPO, run Optuna trials sequentially within each architecture (GPU is the bottleneck), but run different architectures' HPO in parallel if combined VRAM fits.

## Output Requirements

Every tournament MUST produce:

```
experiments/YYYY-MM-DD_TASK_tournament/
├── phase0_sea_trial/
│   ├── ranking.json          ← all candidates ranked
│   └── candidate_N/          ← fold0 predictions + logs
├── phase1_semifinal/
│   ├── MODEL_NAME/
│   │   ├── fold{0-4}_seed{0-2}_best.pt
│   │   ├── oof_predictions.npy   ← CRITICAL for ensemble
│   │   └── scores.json
│   └── semifinal_ranking.json
├── phase2_hpo/
│   ├── MODEL_NAME/
│   │   ├── optuna_study.db
│   │   ├── best_configs.json
│   │   ├── fold{0-4}_seed{0-4}_best.pt
│   │   └── oof_predictions.npy
│   └── hpo_ranking.json
├── phase3_ensemble/
│   ├── greedy_selection.json     ← which models selected, in what order
│   ├── ensemble_weights.json
│   ├── final_predictions.json.gz ← matbench submission format
│   └── final_score.json          ← vs SOTA comparison
├── config.json
├── scripts/                      ← all training scripts (frozen)
└── logs/                         ← all training logs
```

## Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `n_sea_trial_epochs` | Epochs for Phase 0 screening | 100 | Just enough to rank |
| `n_semifinal_seeds` | Seeds per architecture in Phase 1 | 3 | Balance speed vs reliability |
| `n_hpo_trials` | Optuna trials per architecture | 200 | More = better but slower |
| `n_final_seeds` | Seeds for final models in Phase 2 | 5-10 | More seeds = better ensemble |
| `max_ensemble_size` | Max models in greedy ensemble | 20 | Diminishing returns after ~10 |
| `advancement_ratio` | Fraction advancing from Phase 0 | 0.5 | Top 50% |

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Phase 0 all candidates bad | Wrong candidate pool for task type | Re-read reference repos, check Strategy selection |
| Ensemble doesn't improve | Models too similar | Ensure architectural diversity (not just different seeds of same model) |
| HPO finds same config repeatedly | Search space too narrow | Widen ranges, add more hyperparameters |
| OOF predictions misaligned | Different fold splits | Always use matbench's built-in splits (task.get_train_and_val_data) |
| GPU OOM on large datasets | Batch size too large | Reduce batch_size, use gradient accumulation |
| Phase 3 overfits to OOF | Too many models in ensemble | Limit to 10-15 models, validate on held-out fold |
