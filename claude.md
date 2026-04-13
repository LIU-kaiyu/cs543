# AGENT.md — MiDaS Failure Analysis / RoboDepth Benchmark Plan

## 1. Mission
This project builds a reproducible benchmark harness around **MiDaS** for **single-image monocular depth estimation**, with the initial goal of **finding and documenting failure cases** rather than training a new model.

The first phase focuses on **RoboDepth-style corruption robustness analysis**, then expands to cross-dataset checks on **KITTI**, **NYU Depth V2**, and **DIODE**.

---

## 2. Current repository state
The required upstream repositories have already been cloned and placed under `third_party/`:

- `third_party/MiDaS/`
- `third_party/RoboDepth/`

These repositories should remain treated as upstream dependencies rather than as the main project codebase.

---

## 3. Core decisions

### Decision A — MiDaS is the model backend, not the project skeleton
The official **MiDaS** repository should be used as the **model and inference backend**.
The benchmark project should **not** directly modify the upstream MiDaS code as its main working structure.

Instead, the recommended architecture is:
- keep **MiDaS** in `third_party/MiDaS/`
- keep **RoboDepth** in `third_party/RoboDepth/`
- place benchmark logic, wrappers, manifests, metrics, and notebooks in the project’s own directories such as `src/`, `scripts/`, and `notebooks/`

This keeps the codebase cleaner:
- upstream code remains isolated
- local benchmark logic remains controlled by the team
- cloud migration becomes easier
- merge conflicts are reduced

### Decision B — RoboDepth should be treated as a benchmark ecosystem, not as a single ordinary dataset
RoboDepth is best understood as a **robustness benchmark built around corrupted depth-evaluation sets**, especially **KITTI-C** and **NYUDepth2-C**, rather than as one standalone raw dataset.

So when the project refers to “starting with RoboDepth,” the practical interpretation should be:
1. start with **KITTI-C** for outdoor corruption robustness
2. then move to **NYUDepth2-C** for indoor corruption robustness

### Decision C — Start with one practical MiDaS model
The first model to benchmark should be:
- **Primary model:** `dpt_hybrid_384`

A second faster model can be added later for debugging and speed comparison:
- **Secondary baseline:** `midas_v21_small_256`

Rationale:
- `dpt_hybrid_384` offers a good quality-versus-compute balance
- `midas_v21_small_256` is useful for smoke tests and cloud debugging
- larger MiDaS variants should wait until the pipeline is stable

### Decision D — Separate smoke-test data from real benchmark data
A very small test set should be used first to verify that the full pipeline works. Full benchmark datasets should only be introduced after model loading, prediction export, and evaluation logic are already stable.

---

## 4. Repository usage decision
Since the repositories are already cloned, the question is no longer **which repositories to clone**, but **which repository should be treated as primary**.

### Primary upstream dependency
- `third_party/MiDaS/`

Use this repository for:
- official MiDaS inference code
- supported model types
- official preprocessing and model loading behavior
- reproducible access to MiDaS weights

### Secondary reference dependency
- `third_party/RoboDepth/`

Use this repository for:
- corruption benchmark definitions
- evaluation conventions
- reference structure for KITTI-C / NYUDepth2-C style testing
- robustness metrics and benchmark design ideas

### Project-owned codebase
The actual benchmark project should live outside those repositories and wrap them.

Recommended rule:
- **MiDaS runs the model**
- **RoboDepth defines the benchmark context**
- **the project’s own code orchestrates everything**

---

## 5. Dataset download order

### Phase 0 — Smoke test only
The first objective is to confirm that inference works locally and can later be transferred to Colab or ICRN.

#### 5.1 Small RGB image folder
Use one of the following:
- sample images already shipped with MiDaS
- a manually collected folder of 10–20 RGB images

Purpose:
- verify model loading
- verify notebook and script execution
- verify output saving
- verify GPU usage when available

#### 5.2 Optional RoboDepth mini package
If there is a small public evaluation package available from the RoboDepth side, it can be used as an intermediate smoke test.

Purpose:
- verify corruption label parsing
- verify evaluation-loop structure
- verify expected directory assumptions

### Phase 1 — First real benchmark
#### 5.3 KITTI + KITTI-C
This should be the **first main benchmark download**.

Download:
- clean KITTI benchmark data needed for evaluation
- **KITTI-C** from the RoboDepth ecosystem

Why this comes first:
- it is the most natural outdoor corruption benchmark for RoboDepth-style testing
- it is well suited for identifying where MiDaS breaks in driving scenes
- it supports corruption-by-corruption error slicing

Initial uses:
- corruption-type ranking
- severity-level error curves
- qualitative failure galleries
- first end-to-end benchmark report

### Phase 2 — Indoor counterpart
#### 5.4 NYU Depth V2 + NYUDepth2-C
This should be downloaded after KITTI-C is working.

Download:
- official labeled NYU Depth V2 evaluation set
- **NYUDepth2-C** from the RoboDepth ecosystem

Why second:
- it adds indoor corruption robustness
- it reveals whether failure patterns are domain-specific
- it complements KITTI-C cleanly

### Phase 3 — Cross-dataset generalization
#### 5.5 DIODE validation split only
Only the **validation split** should be downloaded initially.

Why validation only:
- the full training split is unnecessarily large for the current goal
- the project is currently about failure analysis and benchmarking, not retraining
- the validation split is enough for error slicing and qualitative inspection

Why DIODE matters:
- it includes both indoor and outdoor scenes
- it helps test generalization outside the RoboDepth corruption setting
- it is useful for finding structural failure cases rather than only corruption-induced ones

---

## 6. Datasets that should not be downloaded first
The following should be avoided during the first setup stage:
- full DIODE training split
- NYU raw unlabeled full archive
- every KITTI-related archive at once
- all datasets mentioned in RoboDepth documentation immediately

Reason:
- storage pressure increases quickly
- cloud setup becomes harder
- debugging becomes slower
- this does not align with the immediate goal of “break MiDaS and identify failure cases”

---

## 7. Recommended project structure

```text
midas-failure-benchmark/
├─ AGENT.md
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ configs/
│  ├─ dataset_paths.yaml
│  ├─ midas_models.yaml
│  └─ eval.yaml
├─ third_party/
│  ├─ MiDaS/
│  └─ RoboDepth/
├─ notebooks/
│  ├─ 00_smoke_test_midas.ipynb
│  ├─ 01_build_kittic_manifest.ipynb
│  ├─ 02_run_midas_on_kittic.ipynb
│  ├─ 03_eval_metrics.ipynb
│  └─ 04_failure_case_analysis.ipynb
├─ src/
│  ├─ adapters/
│  │  └─ midas_adapter.py
│  ├─ datasets/
│  │  ├─ kitti_c.py
│  │  ├─ nyu_c.py
│  │  ├─ diode.py
│  │  └─ transforms.py
│  ├─ evaluation/
│  │  ├─ align.py
│  │  ├─ metrics.py
│  │  └─ robodepth_metrics.py
│  ├─ analysis/
│  │  ├─ failure_slices.py
│  │  └─ report_tables.py
│  └─ utils/
│     ├─ io.py
│     ├─ paths.py
│     └─ seed.py
├─ scripts/
│  ├─ setup_local.sh
│  ├─ download_weights.sh
│  ├─ run_smoke_test.sh
│  ├─ run_kittic_batch.sh
│  ├─ run_nyuc_batch.sh
│  ├─ run_diode_val.sh
│  └─ export_failure_gallery.sh
├─ data/
│  ├─ raw/
│  ├─ manifests/
│  ├─ cache/
│  └─ processed/
└─ outputs/
   ├─ predictions/
   ├─ metrics/
   ├─ tables/
   └─ galleries/
```

---

## 8. Environment recommendation

### Base recommendation
Use **Python 3.10** as the common denominator.

### Reasoning
Python 3.10 is a practical middle ground for:
- local machines
- Google Colab
- ICRN notebook environments
- MiDaS-related dependencies

### Minimum package set
The environment should include at least:
- `torch`
- `torchvision`
- `timm`
- `opencv-python`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tqdm`
- `pyyaml`
- `jupyter`
- `ipywidgets`
- `pillow`

Optional but useful:
- `einops`
- `scikit-image`
- `scikit-learn`
- `tabulate`

### Practical rule
The project should keep its own `environment.yml` or `requirements.txt` and should not depend entirely on the upstream MiDaS environment file.

---

## 9. Important evaluation caveat
MiDaS predicts **relative inverse depth**, not calibrated metric depth.

That means datasets such as KITTI, NYUv2, and DIODE cannot be evaluated correctly by treating raw MiDaS outputs as directly comparable metric depth maps.

### Required evaluation step
For each image, the pipeline should apply an alignment step before computing metric depth errors:
- convert MiDaS output into a depth-compatible representation if needed
- apply **scale** or **scale-and-shift alignment** on valid ground-truth pixels
- then compute benchmark metrics

If alignment is skipped, the evaluation will partly measure scale mismatch rather than true scene understanding.

This is one of the most important methodological constraints in the project.

---

## 10. What should be benchmarked first

### Stage A — Pipeline validation
Run MiDaS on:
- 10–20 random RGB images
- a small corrupted subset if available

Confirm:
- model loads correctly
- output tensor shapes are valid
- outputs are saved correctly
- batch inference works
- GPU is used when available

### Stage B — Main initial benchmark
Run on **KITTI-C** with:
- primary model: `dpt_hybrid_384`
- optional faster baseline later: `midas_v21_small_256`

Record for each corruption type and severity:
- Abs Rel
- δ1
- DEE2
- runtime per image
- failed reads or skipped samples

### Stage C — Failure discovery
For each corruption type, save:
- top 20 worst images
- median 20 images
- best 20 images

This ensures that the project produces interpretable failure evidence rather than only aggregate metrics.

### Stage D — Domain transfer
Repeat the analysis on:
- NYUDepth2-C
- DIODE validation

Then compare:
- outdoor corruption failure patterns
- indoor corruption failure patterns
- clean-domain shift failures

---

## 11. Failure categories to inspect explicitly
The analysis should not stop at one aggregate score. It should create slices.

Important buckets include:
- low light / dark scenes
- fog / haze / reduced contrast
- motion blur / defocus / zoom blur
- sensor noise
- glass / reflective surfaces
- sky-heavy scenes
- textureless walls or roads
- thin structures such as poles, rails, or branches
- transparent or semi-transparent objects
- long-range outdoor scenes
- indoor clutter and occlusion boundaries

The underlying objective is to determine whether MiDaS fails mainly because of:
1. photometric corruption
2. geometry ambiguity
3. depth-range calibration mismatch
4. domain shift
5. structural scene complexity

---

## 12. What to store for every prediction
For every processed image, the benchmark table should store:
- input image path
- dataset name
- split name
- corruption type
- severity
- model name
- raw prediction path
- aligned prediction path
- metric values
- runtime

All records should be written into one central CSV or parquet file.
This file becomes the project’s experimental log.

---

## 13. Local-machine setup plan

### Step 1
Verify that the project root contains the expected folders and that the upstream repositories exist under `third_party/`.

### Step 2
Create the project environment and verify:
- `import torch`
- `import timm`
- `torch.cuda.is_available()`

### Step 3
Run official MiDaS inference once through the upstream CLI.

Example:
```bash
cd third_party/MiDaS
python run.py --model_type dpt_hybrid_384 --input_path input --output_path output
```

### Step 4
Build a project-owned wrapper such as `src/adapters/midas_adapter.py` so that the benchmark code can:
- load a MiDaS model by config name
- run inference over a list of file paths
- save predictions in a consistent output format

### Step 5
Create dataset manifests.
Each row should contain at least:
- image path
- ground-truth depth path
- valid mask path if available
- corruption type
- severity
- split

### Step 6
Run a small 20-image benchmark locally.
Only after this succeeds should the project be uploaded or synced to Colab or ICRN.

---

## 14. Colab / ICRN operating strategy
These environments are session-based and sometimes temporary, so the codebase should assume interruptions.

### Persist the following
- source code
- notebooks
- manifest files
- exported outputs
- cached model weights when possible

### Avoid recomputing expensive work
Cache:
- model weights
- dataset manifests
- finished prediction files
- processed sample lists

### Recommended notebook split
Use separate notebooks for:
1. setup and environment checks
2. manifest creation
3. inference
4. metric computation
5. failure visualization

This is more stable than one monolithic notebook.

### Cloud execution rule
On Colab or ICRN, the benchmark should never begin with the full dataset.
Instead, it should progress in this order:
- 10 images
- 100 images
- one corruption family
- full batch

This avoids wasting a full session on a path or dependency error.

---

## 15. Concrete priority order

### Priority 1
- verify MiDaS runs locally
- verify RoboDepth folder structure and expected benchmark assets
- create the wrapper project structure
- decide one output prediction format, preferably `.npy`

### Priority 2
- build KITTI-C manifest
- run `dpt_hybrid_384` on a small KITTI-C subset
- implement alignment and metric computation
- export first failure gallery

### Priority 3
- expand to full KITTI-C
- add `midas_v21_small_256`
- compare speed versus quality

### Priority 4
- add NYUDepth2-C
- compare indoor versus outdoor corruption failure patterns

### Priority 5
- add DIODE validation
- test cross-dataset behavior beyond RoboDepth corruptions

---

## 16. What should not be done first
The following tasks should be postponed:
- retraining MiDaS
- adding many competitor models before MiDaS works cleanly
- downloading every possible dataset split immediately
- mixing local, Colab, and ICRN path logic inside one fragile notebook
- computing metrics on raw MiDaS output without alignment

These are the most common ways to slow the project down early.

---

## 17. Minimal command checklist

### Environment
```bash
conda create -n midas-benchmark python=3.10 -y
conda activate midas-benchmark
pip install torch torchvision timm opencv-python numpy scipy matplotlib pandas tqdm pyyaml jupyter ipywidgets pillow
```

### Upstream MiDaS smoke test
```bash
cd third_party/MiDaS
python run.py --model_type dpt_hybrid_384 --input_path input --output_path output
```

---

## 18. Final recommendation summary

### Repositories already in place
- `third_party/MiDaS`
- `third_party/RoboDepth`

### Download first
1. tiny smoke-test RGB image set
2. optional small RoboDepth evaluation package if available
3. **KITTI + KITTI-C**
4. **NYU Depth V2 + NYUDepth2-C**
5. **DIODE validation split only**

### Start with model
- `dpt_hybrid_384`

### Add later
- `midas_v21_small_256`

### Main scientific objective
The benchmark should identify **which corruption types, scene types, and structural situations cause MiDaS to fail**, then turn those failures into a structured empirical analysis.
