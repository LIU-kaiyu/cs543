# MiDaS Failure Analysis Benchmark

A reproducible benchmark harness for **single-image monocular depth estimation** using [MiDaS](https://github.com/isl-org/MiDaS), focused on finding and documenting failure cases under real-world corruptions ([RoboDepth](https://github.com/ldkong1205/RoboDepth) benchmark style).

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Repository Layout](#repository-layout)
3. [Quick Start — Local](#quick-start--local)
4. [Quick Start — Google Colab](#quick-start--google-colab)
5. [Notebooks — What Each One Does](#notebooks--what-each-one-does)
6. [Scripts — What Each One Does](#scripts--what-each-one-does)
7. [Source Modules](#source-modules)
8. [Dataset Download Order](#dataset-download-order)
9. [Important: Alignment Before Metrics](#important-alignment-before-metrics)
10. [Config Files](#config-files)

---

## Project Goal

MiDaS predicts **relative inverse depth** — not calibrated metric depth. This project:

1. Wraps MiDaS inference cleanly so it can be called from notebooks and scripts
2. Applies correct scale-and-shift alignment before computing metrics
3. Runs MiDaS over corrupted depth benchmarks (KITTI-C, NYUDepth2-C, DIODE)
4. Slices results by corruption type and severity to pinpoint where MiDaS fails
5. Exports failure galleries (worst / median / best images per corruption)

**Primary model:** `dpt_hybrid_384`  
**Secondary (smoke tests / speed baseline):** `midas_v21_small_256`

---

## Repository Layout

```
.
├── configs/
│   ├── dataset_paths.yaml     # Root paths for all datasets and output dirs
│   ├── midas_models.yaml      # Model names, weight paths, input sizes
│   └── eval.yaml              # Alignment method, metric list, depth ranges
│
├── notebooks/
│   ├── 00_smoke_test_midas.ipynb        # START HERE — verify everything works
│   ├── 01_build_kittic_manifest.ipynb   # Build KITTI-C CSV manifest
│   ├── 02_run_midas_on_kittic.ipynb     # Batch inference → .npy predictions
│   ├── 03_eval_metrics.ipynb            # Alignment + metrics → results CSV
│   ├── 04_failure_case_analysis.ipynb   # Tables, severity curves, galleries
│   ├── 05_build_diode_manifest.ipynb    # Build DIODE val CSV manifest
│   ├── 06_run_midas_on_diode.ipynb      # Batch inference on DIODE val
│   ├── 07_eval_diode_metrics.ipynb      # Alignment + metrics for DIODE
│   └── 08_diode_failure_case_analysis.ipynb  # Domain-wise DIODE analysis
│
├── scripts/
│   ├── setup_local.sh          # Create conda environment
│   ├── setup_third_party.sh    # Clone the upstream MiDaS repo
│   ├── download_weights.sh     # Download dpt_hybrid_384 weights
│   ├── download_diode_val.sh   # Download the DIODE validation split
│   ├── run_smoke_test.sh       # CLI smoke test (no notebook needed)
│   ├── run_kittic_batch.sh     # Full KITTI-C inference + eval in one pass
│   ├── run_diode_batch.sh      # Full DIODE val inference + eval in one pass
│   ├── export_failure_gallery.sh  # Export PNG galleries from results CSV
│   ├── export_diode_gallery.sh   # Export DIODE indoor/outdoor galleries
│   └── export_diode_failure_panels.sh  # Export report-ready DIODE failure panels
│
├── src/
│   ├── adapters/
│   │   └── midas_adapter.py   # MiDaSAdapter — load model, run inference
│   ├── datasets/
│   │   ├── kitti_c.py         # Manifest builder + KittiCDataset
│   │   ├── diode.py           # Manifest builder + DIODEDataset
│   │   └── transforms.py      # Depth PNG loaders (KITTI, NYU, DIODE)
│   ├── evaluation/
│   │   ├── align.py           # Scale-shift alignment (REQUIRED before metrics)
│   │   ├── metrics.py         # abs_rel, rmse, δ1/δ2/δ3, etc.
│   │   └── robodepth_metrics.py  # RoboDepth-compatible eval loop
│   ├── analysis/
│   │   ├── failure_slices.py  # get_worst_n / get_best_n / get_median_n
│   │   └── report_tables.py   # Summary tables and severity curves
│   └── utils/
│       ├── io.py              # Save/load .npy, CSV, parquet
│       ├── paths.py           # Path resolution from config YAML
│       └── seed.py            # set_seed() for reproducibility
│
├── third_party/
│   ├── MiDaS/                 # Upstream MiDaS repo (do not modify)
│   └── RoboDepth/             # Upstream RoboDepth repo (do not modify)
│
├── data/
│   ├── raw/                   # Downloaded dataset files go here
│   ├── manifests/             # Generated CSV manifests go here
│   ├── cache/                 # Cached intermediates
│   └── processed/             # Any preprocessed outputs
│
├── outputs/
│   ├── predictions/           # .npy depth prediction files
│   ├── metrics/               # kittic_results.csv and similar
│   ├── tables/                # Exported summary tables
│   └── galleries/             # PNG failure gallery images
│
├── environment.yml
└── requirements.txt
```

---

## Quick Start — Local

### Step 1 — Create the environment

```bash
conda create -n midas-benchmark python=3.10 -y
conda activate midas-benchmark
pip install -r requirements.txt
```

Or use the provided script:

```bash
bash scripts/setup_local.sh
conda activate midas-benchmark
```

### Step 2 — Clone MiDaS and download model weights

```bash
bash scripts/setup_third_party.sh
bash scripts/download_weights.sh
```

This downloads `dpt_hybrid_384.pt` into `third_party/MiDaS/weights/`. The file is ~470 MB.

### Step 3 — Run the smoke test

MiDaS ships with a few sample images in `third_party/MiDaS/input/`. Use these to verify everything works before touching any benchmark data.

```bash
# Option A: from a notebook
jupyter notebook notebooks/00_smoke_test_midas.ipynb

# Option B: from the terminal
bash scripts/run_smoke_test.sh
```

Expected output: `.npy` depth files appear in `outputs/predictions/smoke/` and the notebook shows depth visualisations.

### Step 4 — Download benchmark data (see [Dataset Download Order](#dataset-download-order))

### Step 5 — Run notebooks in order: 01 → 02 → 03 → 04

---

## Quick Start — Google Colab

Because Colab sessions are temporary, keep the source code and outputs on **Google Drive** and mount it at the start of every session.

### One-time setup (run once, then reuse)

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Clone or copy the project into Drive
import os
PROJECT = '/content/drive/MyDrive/cs543-midas'
os.makedirs(PROJECT, exist_ok=True)
# If you have the project on GitHub:
# !git clone https://github.com/YOUR_REPO.git {PROJECT}
```

```python
# Cell 3 — Install dependencies (do this every session)
%pip install -q timm opencv-python-headless scipy matplotlib pandas tqdm pyyaml ipywidgets pillow einops scikit-image tabulate
```

```python
# Cell 4 — Add project AND MiDaS to sys.path (do this every session)
import sys
sys.path.insert(0, PROJECT)
sys.path.insert(0, f'{PROJECT}/third_party/MiDaS')
# Both lines are required: the first lets Python find src/, the second lets
# midas_adapter.py locate the midas package on Drive's FUSE filesystem.
```

```python
# Cell 5 — Download weights if not already on Drive (one-time)
import os
weights_dir = f'{PROJECT}/third_party/MiDaS/weights'
os.makedirs(weights_dir, exist_ok=True)
if not os.path.exists(f'{weights_dir}/dpt_hybrid_384.pt'):
    !wget -q -O {weights_dir}/dpt_hybrid_384.pt \
        https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt
    print('Downloaded.')
else:
    print('Weights already present.')
```

### Every session: update dataset paths

Edit `configs/dataset_paths.yaml` to point `kitti_c.root` at your Drive path, e.g.:

```yaml
kitti_c:
  root: /content/drive/MyDrive/cs543-midas/data/raw/kitti_c
  gt_path: /content/drive/MyDrive/cs543-midas/data/raw/kitti_gt
```

### Colab execution order

Start small — Colab sessions have time limits and OOM risks.

| Step | Action |
|------|--------|
| 1 | Run `00_smoke_test_midas.ipynb` on the MiDaS sample images |
| 2 | Set `MAX_SAMPLES = 10` in notebook 02, verify the loop works |
| 3 | Increase to `MAX_SAMPLES = 100`, confirm outputs are saved to Drive |
| 4 | Set `MAX_SAMPLES = None` for a full corruption family |
| 5 | Run full batch only after the pipeline is confirmed stable |

**Never start with the full dataset on Colab.** One path error wastes the whole session.

### Enabling GPU on Colab

`Runtime → Change runtime type → T4 GPU`

The adapter auto-detects CUDA:

```python
from src.adapters.midas_adapter import MiDaSAdapter
adapter = MiDaSAdapter(model_type='dpt_hybrid_384')  # will use GPU automatically
```

---

## Notebooks — What Each One Does

### `00_smoke_test_midas.ipynb` ← **Start here**

Verifies the full pipeline on the MiDaS built-in sample images. No benchmark data needed.

- Checks PyTorch version and CUDA availability
- Loads `dpt_hybrid_384`
- Runs inference on every image in `third_party/MiDaS/input/`
- Saves `.npy` predictions to `outputs/predictions/smoke/`
- Displays side-by-side RGB + depth map

**If this fails, fix it before touching anything else.**

---

### `01_build_kittic_manifest.ipynb`

Scans your KITTI-C directory and builds a CSV manifest used by all later steps.

- Reads `configs/dataset_paths.yaml` for the KITTI-C root path
- Produces `data/manifests/kitti_c_manifest.csv`
- Columns: `image_path, gt_path, corruption_type, severity, sequence, frame_id, split`

**Requires:** KITTI-C data downloaded and `dataset_paths.yaml` updated.

---

### `02_run_midas_on_kittic.ipynb`

Batch inference over the manifest. Skips images that already have a saved `.npy`.

- Set `MAX_SAMPLES = 50` to test on a small subset first
- Saves one `.npy` per image under `outputs/predictions/kitti_c/<corruption>/<severity>/`
- Resume-safe: re-running skips already-completed predictions

---

### `03_eval_metrics.ipynb`

Loads each `.npy` prediction + its ground-truth depth PNG, applies alignment, computes metrics.

- **Applies `align_scale_shift` before every metric** — this is required
- Computes: `abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3`
- Saves results to `outputs/metrics/kittic_results.csv`
- Prints a per-corruption summary table at the end

---

### `04_failure_case_analysis.ipynb`

Loads `kittic_results.csv` and produces failure analysis.

- Per-corruption mean metric table
- Corruption × severity pivot table
- Severity curves (abs_rel vs. severity per corruption type)
- Gallery: worst 4 images for the hardest corruption type

---

### `05_build_diode_manifest.ipynb`

Scans the DIODE validation split and builds a CSV manifest.

- Reads `configs/dataset_paths.yaml` for `diode.val_root`
- Produces `data/manifests/diode_val_manifest.csv`
- Adds `domain` metadata (`indoors` / `outdoor`) plus reusable analysis columns

### `06_run_midas_on_diode.ipynb`

Batch inference over the DIODE validation manifest.

- Saves one `.npy` per image under `outputs/predictions/diode/<domain>/<scene>/<scan>/`
- Resume-safe: re-running skips already-completed predictions
- Supports a small `MAX_SAMPLES` smoke run before the full split

### `07_eval_diode_metrics.ipynb`

Evaluates MiDaS predictions against DIODE depth + validity masks.

- Loads DIODE `*_depth.npy` and `*_depth_mask.npy`
- **Applies `align_scale_shift` before every metric** — this is required
- Saves results to `outputs/metrics/diode_results.csv`
- Reports both overall and indoor/outdoor summaries

### `08_diode_failure_case_analysis.ipynb`

Summarises DIODE failure patterns and exports domain-wise slices.

- Overall metrics and `domain` breakdown
- Worst / median / best samples for `indoors` and `outdoor`
- Ready-made tables for the midterm report

---

## Scripts — What Each One Does

| Script | Purpose |
|--------|---------|
| `setup_local.sh` | Create conda env and install packages |
| `setup_third_party.sh` | Clone the upstream MiDaS repo into `third_party/MiDaS` |
| `download_weights.sh` | Download `dpt_hybrid_384.pt` weights (~470 MB) |
| `download_diode_val.sh` | Download and extract the official DIODE validation split |
| `run_smoke_test.sh` | CLI equivalent of notebook 00 — no Jupyter needed |
| `run_kittic_batch.sh` | Full KITTI-C inference + alignment + metrics in one pass |
| `run_diode_batch.sh` | Full DIODE inference + alignment + metrics in one pass |
| `export_failure_gallery.sh` | Export worst/median/best PNGs for every corruption type |
| `export_diode_gallery.sh` | Export worst/median/best PNGs for DIODE indoor/outdoor domains |
| `export_diode_failure_panels.sh` | Export report-ready DIODE top-failure figures with RGB / GT / prediction / error |

Run scripts from the project root:

```bash
cd "C:/CS543 Project"
bash scripts/download_weights.sh
bash scripts/run_smoke_test.sh
```

---

## Source Modules

### `src/adapters/midas_adapter.py`

The main inference interface. Import this instead of calling MiDaS directly.

```python
from src.adapters.midas_adapter import MiDaSAdapter

adapter = MiDaSAdapter(model_type='dpt_hybrid_384')

# Single image → numpy depth map
depth = adapter.predict('path/to/image.jpg')

# Batch → list of result dicts with pred_path and runtime_s
records = adapter.run_batch(image_paths, output_dir='outputs/predictions/smoke')
```

### `src/evaluation/align.py`

**Must be called before any metric.** MiDaS output is relative inverse depth — not metres.

```python
from src.evaluation.align import align_scale_shift

aligned = align_scale_shift(pred, gt, valid_mask)  # least-squares scale + shift
```

### `src/evaluation/metrics.py`

```python
from src.evaluation.metrics import compute_all_metrics

m = compute_all_metrics(aligned, gt, valid_mask, min_depth=1e-3, max_depth=80.0)
# m = {'abs_rel': ..., 'rmse': ..., 'delta1': ..., ...}
```

### `src/datasets/kitti_c.py`

```python
from src.datasets.kitti_c import build_manifest, KittiCDataset

df = build_manifest(kitti_c_root, gt_root)          # returns DataFrame
dataset = KittiCDataset(df, corruption_filter='fog') # PyTorch Dataset
```

### `src/datasets/diode.py`

```python
from src.datasets.diode import build_manifest, DIODEDataset

df = build_manifest(diode_val_root)               # returns DataFrame
dataset = DIODEDataset(df, domain_filter='outdoor')
```

### `src/analysis/failure_slices.py`

```python
from src.analysis.failure_slices import get_worst_n

worst = get_worst_n(results_df, metric='abs_rel', corruption_type='fog', n=20)
```

### `src/analysis/report_tables.py`

```python
from src.analysis.report_tables import corruption_summary_table, severity_curve

corruption_summary_table(df)          # mean metrics per corruption type
severity_curve(df, 'motion_blur')     # abs_rel vs severity for one corruption
```

---

## Dataset Download Order

Follow this order. Do not download everything at once.

| Phase | Dataset | Why |
|-------|---------|-----|
| 0 | MiDaS sample images (already in `third_party/MiDaS/input/`) | Smoke test |
| 1 | **KITTI + KITTI-C** | First real benchmark — outdoor corruption robustness |
| 2 | **NYU Depth V2 + NYUDepth2-C** | Indoor counterpart — run after KITTI-C works |
| 3 | **DIODE validation split only** | Cross-dataset generalisation check |

Update `configs/dataset_paths.yaml` with the actual paths after each download.

---

## Important: Alignment Before Metrics

MiDaS predicts **relative inverse depth**, not metric depth in metres. Computing metrics directly on raw MiDaS output will measure scale mismatch, not depth quality.

The pipeline always does:

```
raw MiDaS output
      ↓
align_scale_shift(pred, gt, valid_mask)   ← least-squares fit on valid GT pixels
      ↓
compute_all_metrics(aligned, gt, ...)     ← now in comparable units
```

See `src/evaluation/align.py` for implementation details.

---

## Config Files

### `configs/dataset_paths.yaml`

Update the root paths to match your local or Drive layout before running any benchmark notebook.

```yaml
kitti_c:
  root: data/raw/kitti_c       # ← change this
  gt_path: data/raw/kitti_gt   # ← change this
```

Relative paths are resolved from the project root automatically via `src/utils/paths.py`.

### `configs/midas_models.yaml`

Lists available models. The default is `dpt_hybrid_384`. Add `midas_v21_small_256` later for speed comparisons.

### `configs/eval.yaml`

Controls alignment method (`scale_shift` or `scale_only`), metric list, and per-dataset depth ranges (KITTI: 0–80 m, NYU: 0–10 m, DIODE: 0–300 m).
