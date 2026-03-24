# MatBench v0.1 SOTA Leaderboard Reference

## When to Use

- Before reporting benchmark results — **always compare against this reference, not from memory**
- When deciding which model architecture to reproduce or improve upon
- When writing result comparison tables or reports to the user
- When planning which tasks to prioritize for SOTA attempts

## CRITICAL RULE

**NEVER cite SOTA numbers from memory or estimation. Always read this file to get the correct SOTA values.** Incorrect SOTA comparisons (e.g., claiming to beat SOTA when comparing against a weaker model) are misleading and unacceptable.

## Cross-Validation Scheme

- **5-fold nested cross-validation** with fixed train/test splits per fold
- All participants use identical splits for fair comparison
- Results reported as mean ± std across 5 folds

## Evaluation Metrics

### Regression Tasks (10 tasks)

| Metric | Description | Ranking |
|--------|-------------|---------|
| **Mean MAE** | Mean Absolute Error averaged over 5 folds | **PRIMARY (↓ lower = better)** |
| Std MAE | Standard deviation of MAE across folds | Secondary (stability) |
| Mean RMSE | Root Mean Squared Error averaged over 5 folds | Secondary |
| Max Max_Error | Largest single-point prediction error across all folds | Secondary |

### Classification Tasks (3 tasks)

| Metric | Description | Ranking |
|--------|-------------|---------|
| **Mean ROC-AUC** | Area Under ROC Curve averaged over 5 folds | **PRIMARY (↑ higher = better)** |
| Std ROC-AUC | Standard deviation of ROC-AUC across folds | Secondary (stability) |
| Mean F1 | F1 score averaged over 5 folds | Secondary |
| Mean Balanced Accuracy | Balanced accuracy averaged over 5 folds | Secondary |

---

## SOTA Leaderboard: All 13 Tasks

### Task 1: matbench_steels

- **Samples:** 312
- **Input:** Composition (chemical formula)
- **Property:** Yield strength of steels
- **Units:** MPa
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **TPOT-Mat** | 79.95 | — | https://github.com/EpistasisLab/tpot |
| 2 | AutoML-Mat | 82.30 | — | https://github.com/mm-tud/automl-materials |
| 3 | MODNet v0.1.12 | 87.76 | — | https://github.com/ppdebreuck/modnet |

### Task 2: matbench_jdft2d

- **Samples:** 636
- **Input:** Structure (pymatgen Structure)
- **Property:** Exfoliation energy of 2D materials (JARVIS-DFT)
- **Units:** meV/atom
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **MODNet v0.1.12** | 33.19 | — | https://github.com/ppdebreuck/modnet |
| 2 | MODNet v0.1.10 | 34.54 | — | https://github.com/ppdebreuck/modnet |
| 3 | coNGN | 36.17 | — | https://github.com/aimat-lab/gcnn_keras |

### Task 3: matbench_phonons

- **Samples:** 1,265
- **Input:** Structure (pymatgen Structure)
- **Property:** Highest phonon frequency (last peak in phonon DOS)
- **Units:** cm⁻¹
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **MegNet (kgcnn v2.1.0)** | 28.76 | — | https://github.com/aimat-lab/gcnn_keras |
| 2 | coNGN | 28.89 | — | https://github.com/aimat-lab/gcnn_keras |
| 3 | ALIGNN | 29.54 | — | https://github.com/usnistgov/alignn |

### Task 4: matbench_expt_gap

- **Samples:** 4,604
- **Input:** Composition (chemical formula)
- **Property:** Experimental band gap
- **Units:** eV
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **Darwin** | 0.2865 | — | https://github.com/hitarth64/DARWIN |
| 2 | Ax/SAASBO CrabNet v1.2.7 | 0.3310 | — | https://github.com/sparks-baird/crabnet-hyperparameter |
| 3 | MODNet v0.1.12 | 0.3327 | — | https://github.com/ppdebreuck/modnet |

### Task 5: matbench_dielectric

- **Samples:** 4,764
- **Input:** Structure (pymatgen Structure)
- **Property:** Refractive index (electronic contribution to dielectric constant)
- **Units:** Dimensionless
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **MODNet v0.1.12** | 0.2711 | — | https://github.com/ppdebreuck/modnet |
| 2 | MODNet v0.1.10 | 0.2970 | — | https://github.com/ppdebreuck/modnet |
| 3 | coGN | 0.3088 | — | https://github.com/aimat-lab/gcnn_keras |

### Task 6: matbench_expt_is_metal

- **Samples:** 4,921
- **Input:** Composition (chemical formula)
- **Property:** Whether the material is metallic (binary)
- **Units:** N/A (classification)
- **Type:** Classification
- **Primary metric:** ROC-AUC ↑

| Rank | Model | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Repository |
|------|-------|-------------|-------------|---------|------------|
| 1 | **Darwin** | 0.9598 | — | — | https://github.com/hitarth64/DARWIN |
| 2 | AMMExpress v2020 | 0.9209 | — | — | https://github.com/hackingmaterials/automatminer |
| 3 | RF-SCM/Magpie | 0.9167 | — | — | https://github.com/hackingmaterials/matbench |

### Task 7: matbench_glass

- **Samples:** 5,680
- **Input:** Composition (chemical formula)
- **Property:** Whether the composition can form metallic glass (binary)
- **Units:** N/A (classification)
- **Type:** Classification
- **Primary metric:** ROC-AUC ↑

| Rank | Model | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Repository |
|------|-------|-------------|-------------|---------|------------|
| 1 | **MODNet v0.1.12** | 0.9603 | — | — | https://github.com/ppdebreuck/modnet |
| 2 | AMMExpress v2020 | 0.8607 | — | — | https://github.com/hackingmaterials/automatminer |
| 3 | RF-SCM/Magpie | 0.8587 | — | — | https://github.com/hackingmaterials/matbench |

### Task 8: matbench_log_gvrh

- **Samples:** 10,987
- **Input:** Structure (pymatgen Structure)
- **Property:** Log10 of VRH-averaged shear modulus (G_VRH)
- **Units:** log₁₀(GPa)
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **coNGN** | 0.0670 | — | https://github.com/aimat-lab/gcnn_keras |
| 2 | coGN | 0.0689 | — | https://github.com/aimat-lab/gcnn_keras |
| 3 | ALIGNN | 0.0715 | — | https://github.com/usnistgov/alignn |

### Task 9: matbench_log_kvrh

- **Samples:** 10,987
- **Input:** Structure (pymatgen Structure)
- **Property:** Log10 of VRH-averaged bulk modulus (K_VRH)
- **Units:** log₁₀(GPa)
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **coNGN** | 0.0491 | — | https://github.com/aimat-lab/gcnn_keras |
| 2 | coGN | 0.0535 | — | https://github.com/aimat-lab/gcnn_keras |
| 3 | MODNet v0.1.12 | 0.0548 | — | https://github.com/ppdebreuck/modnet |

### Task 10: matbench_perovskites

- **Samples:** 18,928
- **Input:** Structure (pymatgen Structure)
- **Property:** Formation energy of perovskites (DFT-computed)
- **Units:** eV/atom
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **coGN** | 0.0269 | — | https://github.com/aimat-lab/gcnn_keras |
| 2 | ALIGNN | 0.0288 | — | https://github.com/usnistgov/alignn |
| 3 | DeeperGATGNN | 0.0288 | — | https://github.com/usccolumbia/deeperGATGNN |

### Task 11: matbench_mp_gap

- **Samples:** 106,113
- **Input:** Structure (pymatgen Structure)
- **Property:** Band gap (DFT PBE, from Materials Project)
- **Units:** eV
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **coGN** | 0.1559 | — | https://github.com/aimat-lab/gcnn_keras |
| 2 | DeeperGATGNN | 0.1694 | — | https://github.com/usccolumbia/deeperGATGNN |
| 3 | coNGN | 0.1697 | — | https://github.com/aimat-lab/gcnn_keras |

### Task 12: matbench_mp_is_metal

- **Samples:** 106,113
- **Input:** Structure (pymatgen Structure)
- **Property:** Whether the material is metallic (binary, from Materials Project)
- **Units:** N/A (classification)
- **Type:** Classification
- **Primary metric:** ROC-AUC ↑

| Rank | Model | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Repository |
|------|-------|-------------|-------------|---------|------------|
| 1 | **CGCNN v2019** | 0.9520 | 0.0074 | 0.9462 | https://github.com/txie-93/cgcnn |
| 2 | ALIGNN | 0.9128 | 0.0015 | 0.9015 | https://github.com/usnistgov/alignn |
| 3 | coGN | 0.9124 | — | — | https://github.com/aimat-lab/gcnn_keras |

### Task 13: matbench_mp_e_form

- **Samples:** 132,752
- **Input:** Structure (pymatgen Structure)
- **Property:** Formation energy (DFT, from Materials Project)
- **Units:** eV/atom
- **Type:** Regression
- **Primary metric:** MAE ↓

| Rank | Model | Mean MAE | Std MAE | Repository |
|------|-------|----------|---------|------------|
| 1 | **coGN** | 0.0170 | 0.0003 | https://github.com/aimat-lab/gcnn_keras |
| 2 | coNGN | 0.0178 | 0.0004 | https://github.com/aimat-lab/gcnn_keras |
| 3 | ALIGNN | 0.0215 | 0.0005 | https://github.com/usnistgov/alignn |

---

## Model Quick Reference

| Model | #1 Count | Type | Key Strength | Repository |
|-------|----------|------|-------------|------------|
| **coGN** | 4 (perovskites, mp_gap, mp_e_form, dielectric) | Equivariant GNN | Large structure datasets | https://github.com/aimat-lab/gcnn_keras |
| **coNGN** | 2 (log_gvrh, log_kvrh) | Nested Line Graph GNN | Medium structure datasets | https://github.com/aimat-lab/gcnn_keras |
| **MODNet** | 3 (jdft2d, dielectric, glass) | Feature engineering + NN | Small-medium datasets, composition+structure | https://github.com/ppdebreuck/modnet |
| **Darwin** | 2 (expt_gap, expt_is_metal) | Evolutionary + GNN | Composition-only tasks | https://github.com/hitarth64/DARWIN |
| **TPOT-Mat** | 1 (steels) | AutoML (genetic programming) | Very small datasets (<500) | https://github.com/EpistasisLab/tpot |
| **MegNet** | 1 (phonons) | Graph Network | Phonon properties | https://github.com/aimat-lab/gcnn_keras |
| **CGCNN** | 1 (mp_is_metal) | Crystal Graph CNN | Metal classification | https://github.com/txie-93/cgcnn |
| **ALIGNN** | 0 (but top-3 in 7 tasks) | Line Graph GNN | Consistent all-rounder | https://github.com/usnistgov/alignn |
| **DeeperGATGNN** | 0 (but top-3 in 2 tasks) | Deep Graph Attention | Scalable deep GNN | https://github.com/usccolumbia/deeperGATGNN |
| **CrabNet** | 0 (but top-3 in 1 task) | Composition Transformer | Composition-only attention | https://github.com/anthony-wang/CrabNet |
| **AMMExpress** | 0 (but top-3 in 2 tasks) | AutoML pipeline | Baseline reference | https://github.com/hackingmaterials/automatminer |

## Patterns for Choosing Models

```
What input type?
├── Composition only (steels, expt_gap, expt_is_metal, glass)
│   ├── Very small (<500 samples) → TPOT-Mat / AutoML-Mat
│   ├── Small-medium (1k-5k) → Darwin, MODNet, CrabNet
│   └── Ensemble of multiple → Often best approach
│
└── Structure (jdft2d, phonons, dielectric, log_gvrh, log_kvrh,
│              perovskites, mp_gap, mp_is_metal, mp_e_form)
    ├── Small (<2k samples) → MODNet, MegNet
    ├── Medium (2k-20k) → coGN/coNGN, ALIGNN, MODNet
    └── Large (>100k) → coGN/coNGN (dominant), DeeperGATGNN
```

## Data Source

Leaderboard data retrieved from https://matbench.materialsproject.org/Leaderboards%20Per-Task/
Last updated: 2026-03-24
