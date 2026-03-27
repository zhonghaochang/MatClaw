# Reference Code Library for MatBench Models

## When to Use
- You need to study SOTA model architectures before implementing
- You want to reproduce a specific model's approach on matbench
- You need to understand how a model handles crystal graph construction, training loops, or feature engineering
- You want to adapt an existing model's code for a new task

## Location
All reference repositories are at `/workspace/group/matbench/reference/repos/`. These are **read-only references** — never modify them. Copy code to your experiment directory before editing.

## Repository Index

### Tier 1: MatBench SOTA Models (directly relevant)

| Repo | Model | Year | Key Innovation | Best MatBench Tasks | Key Files to Read |
|------|-------|------|----------------|--------------------|--------------------|
| `alignn/` | ALIGNN | 2021 | Line graph for bond angles | mp_gap, mp_e_form | `alignn/models/alignn.py`, `alignn/train.py` |
| `matgl/` | M3GNet, MatGL | 2022-24 | Universal potential, multi-fidelity | All structure tasks | `matgl/models/`, `pretrained/` |
| `mace/` | MACE-MP-0 | 2023 | Higher-order equivariant messages | Transfer learning baseline | `mace/modules/models.py` |
| `chgnet/` | CHGNet | 2023 | Charge-informed GNN | Universal potential | `chgnet/model/`, `chgnet/trainer/` |
| `kgcnn/` | coGN, coNGN, DimeNet++ | 2022-24 | Continuous filter, nested GNN | **Current SOTA on 5+ tasks** | `kgcnn/literature/`, `training/` |
| `modnet/` | MODNet | 2021 | Optimal descriptor selection | dielectric, glass, jdft2d | `modnet/models/`, `modnet/featurizers/` |
| `crabnet/` | CrabNet | 2021 | Attention on elements | Composition tasks | `crabnet/model.py` |
| `roost/` | Roost | 2020 | Set2Set on stoichiometry | Composition tasks | `roost/model.py`, `roost/cgcnn.py` |
| `cgcnn/` | CGCNN | 2018 | Original crystal graph CNN | Baseline for all structure | `cgcnn/model.py`, `cgcnn/data.py` |
| `megnet/` | MEGNet | 2019 | Global state + edge updates | log_gvrh, log_kvrh | `megnet/models/` |
| `darwin/` | Darwin | 2024 | Evolutionary + ensemble | expt_gap, expt_is_metal | Look for training scripts |

### Tier 2: Advanced Equivariant Models

| Repo | Model | Year | Key Innovation | Key Files |
|------|-------|------|----------------|-----------|
| `nequip/` | NequIP | 2022 | E(3)-equivariant interatomic potentials | `nequip/nn/`, `nequip/model/` |
| `allegro/` | Allegro | 2023 | Scalable equivariant, local descriptors | `allegro/model/` |
| `schnetpack/` | SchNet, PaiNN | 2017-21 | Continuous-filter conv, equivariant | `schnetpack/nn/`, `src/schnetpack/` |
| `gemnet/` | GemNet | 2021 | Directional message passing | `gemnet/model/` |
| `fairchem-ocp/` | EquiformerV2, eSCN, SCN | 2023-24 | Equivariant transformers (Meta) | `src/fairchem/core/models/` |
| `torchmd-net/` | TorchMD-NET, ViSNet | 2022-24 | MD-focused equivariant | `torchmdnet/models/` |
| `e3nn-gnn/` | E3NN examples | 2021 | General e3nn graph networks | Examples and tutorials |
| `egnn/` | EGNN | 2021 | E(n) equivariant simple GNN | `models/egnn_clean/` |
| `sevennet/` | SevenNet | 2024 | Samsung universal potential | `sevenn/nn/`, `sevenn/train/` |

### Tier 3: Foundation Models & Universal Potentials (2024-2026)

| Repo | Model | Year | Key Innovation | Key Files |
|------|-------|------|----------------|-----------|
| `orb-models/` | ORB | 2024 | Orbital materials foundation | `orb_models/` |
| `jmp/` | JMP | 2024 | Joint multi-domain pre-training (Meta) | `jmp/models/` |
| `mattersim/` | MatterSim | 2024 | Microsoft universal simulator | Model configs |
| `bamboo/` | BAMBOO | 2024 | Beyond-atomistic model | `bamboo/` |
| `unimol/` | Uni-Mol | 2023 | 3D molecular pre-training (ByteDance) | `unimol/models/` |

### Tier 4: Generative & Crystal Structure Prediction

| Repo | Model | Year | Key Innovation | Key Files |
|------|-------|------|----------------|-----------|
| `cdvae/` | CDVAE | 2022 | Diffusion VAE for crystals | `cdvae/pl_modules/` |
| `diffcsp/` | DiffCSP | 2023 | Diffusion for CSP | `diffcsp/pl_modules/` |
| `mattergen/` | MatterGen | 2024 | Microsoft generative model | `mattergen/` |
| `flowmm/` | FlowMM | 2024 | Flow matching for materials (Meta) | `flowmm/` |

### Tier 5: Benchmarks & Tools

| Repo | Purpose | Key Files |
|------|---------|-----------|
| `matbench/` | Official benchmark code | `matbench/bench.py`, `benchmarks/` (all SOTA submissions!) |
| `matbench-discovery/` | Materials discovery benchmark | `models/`, `data/` |
| `matminer/` | Feature engineering library | `matminer/featurizers/` |
| `automatminer/` | AutoML for materials | `automatminer/` |
| `matsciml/` | Intel's MatSci ML framework | Multi-task, multi-dataset training |
| `dig-models/` | DIG library (SphereNet etc.) | `dig/threedgraph/` |
| `maml/` | Materials ML library (MV Lab) | `maml/apps/` |

## How to Use This Library

### 1. Study a Model Architecture
```bash
# Example: understand how ALIGNN builds line graphs
cat /workspace/group/matbench/reference/repos/alignn/alignn/graphs.py
cat /workspace/group/matbench/reference/repos/alignn/alignn/models/alignn.py
```

### 2. Find Training Configurations
```bash
# Example: see how coGN trains on matbench
find /workspace/group/matbench/reference/repos/kgcnn -name "*.py" | xargs grep -l "matbench"
cat /workspace/group/matbench/reference/repos/kgcnn/training/hyper/hyper_mp_e_form.py
```

### 3. Look at Official MatBench Submissions
```bash
# The matbench repo contains ALL official submissions with code!
ls /workspace/group/matbench/reference/repos/matbench/benchmarks/
# Each folder has: results.json.gz + run.py (or notebook)
cat /workspace/group/matbench/reference/repos/matbench/benchmarks/matbench_v0.1_coGN/run.py
```

### 4. Adapt Code for Your Experiment
```python
# NEVER modify reference repos. Copy to experiment dir first:
import shutil
shutil.copytree(
    "/workspace/group/matbench/reference/repos/alignn/alignn",
    "/workspace/group/matbench/experiments/2026-03-25_alignn_custom/alignn_src"
)
# Then modify the copy
```

## Strategy Guide: Which Model to Try Next

```
Want to beat SOTA on a specific task?
├── mp_e_form, mp_gap, perovskites → Study coGN (kgcnn/) — current king
├── jdft2d, dielectric → Study MODNet (modnet/) — good on small data
├── expt_gap, expt_is_metal → Study Darwin (darwin/) — composition SOTA
├── log_gvrh, log_kvrh → Study coNGN (kgcnn/) — nested GNN
├── glass → Study MODNet (modnet/)
├── steels → Study TPOT/automatminer — AutoML approach
├── phonons → Study MEGNet (megnet/)
└── mp_is_metal → Study CGCNN (cgcnn/) — surprisingly strong

Want to try a new approach not yet on leaderboard?
├── EquiformerV2 (fairchem-ocp/) — best on OC20, not yet submitted to matbench
├── MACE fine-tuning (mace/) — universal potential → task-specific
├── SevenNet (sevennet/) — Samsung's potential, very recent
├── Uni-Mol (unimol/) — 3D pre-training, unexplored on matbench
└── ORB (orb-models/) — orbital materials foundation model
```

## Important Notes
- All repos are **shallow clones** (--depth 1). Use `git fetch --unshallow` if you need full history.
- Repos are on the **data disk** (vepfs), not system disk. No space concerns.
- NEVER `pip install` from these repos directly. Create a conda env if needed (see sota-reproduction skill).
- The `matbench/benchmarks/` directory is gold — it has the actual code used by every leaderboard entry.
