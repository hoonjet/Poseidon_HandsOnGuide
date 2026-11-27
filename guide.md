# Poseidon PDE Finetuning — Hands-On Guide

This guide walks you through **three self-contained finetuning tutorials** built on the
[Poseidon](https://arxiv.org/abs/2405.19101) foundation model family (scOT architecture).
Each tutorial adapts the pretrained **Poseidon-T** (20.77M parameters) to a different
PDE problem using a plain PyTorch training loop — no W&B, no accelerate, no scOT Trainer.

| Tutorial | Script | PDE / Physics | Type | Dataset Size |
|----------|--------|---------------|------|--------------|
| 1. Wave-Layer | `solve_wave_pde_tutorial.py` | Acoustic wave equation in a layered medium — wave field propagation | Time-dependent | ~15 GB |
| 2. Poisson-Gauss | `solve_poisson_gauss_tutorial.py` | Poisson equation ∇²u = f — steady-state elliptic field | Steady | ~2.6 GB |
| 3. SE-AF Airfoil | `solve_airfoil_tutorial.py` | Steady compressible Euler flow over airfoils — density field | Steady | ~1.4 GB |

**What you will learn across the three tutorials:**

- **Time-dependent problems** (Tutorial 1): predicting the next time step u(t) → u(t+Δt)
  with time conditioning (`ConditionalLayerNorm`), cosine LR schedule, and gradient clipping.
- **Steady-state problems** (Tutorials 2 & 3): mapping an input field directly to an
  output field with **no time conditioning** (`use_conditioning=False`).
- **Geometry-conditioned surrogates** (Tutorial 3): the most CAE-like workflow — an
  airfoil shape mask in, steady density field out, i.e. a design-stage aero-solver surrogate.
- **Pixel masking** (Tutorial 3): excluding solid-body pixels from the loss via `pixel_mask`.
- **Parameter-group-aware optimization**: separate learning rates for the backbone
  (5e-5) and the re-initialized embedding/recovery layers (5e-4), mirroring scOT's Trainer.

All datasets are downloaded automatically from the
[HuggingFace Hub](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14)
the first time you run a tutorial.

---

## 1. Environment Setup

### 1.1 Prerequisites

- Python 3.10+
- An NVIDIA GPU with ≥ 8 GB VRAM (the tutorials were verified on a Quadro P4000 8 GB)
- ~25 GB free disk space for all three datasets (or less if you run one tutorial at a time)

### 1.2 Create the environment

```bash
# Clone this repository
git clone <YOUR-REPO-URL> poseidon-tutorials
cd poseidon-tutorials

# Create and activate a virtual environment
python -m venv poseidon_env
# Windows:
poseidon_env\Scripts\activate
# Linux / macOS:
source poseidon_env/bin/activate

# Install PyTorch (pick the right index for your CUDA version — see pytorch.org)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
pip install -r requirements.txt

# Install the scOT package in editable mode
pip install -e .
```

### 1.3 Corporate proxy / SSL note

If you are behind a corporate proxy with SSL interception, the scripts handle it
automatically: they patch `requests` to skip certificate verification, download
datasets via `curl -k`, and disable HDF5 file locking. No manual configuration needed.


---

## 2. Tutorial 1 — Wave-Layer (Acoustic Wave Equation, Time-Dependent)

### Physics

The acoustic wave equation ∂²u/∂t² = c(x,y)·∇²u in a horizontally **layered medium**:
the wave speed c varies stepwise across layers, producing reflection and refraction.
The model sees the wave field u(t=0) and the coefficient field c, and predicts u(t=14).

### Run

```bash
python solve_wave_pde_tutorial.py
```

The script downloads 4 chunked `.nc` files (~15 GB total) and assembles them into a
single logical dataset using an **HDF5 Virtual Dataset (VDS)** — a 6 KB file with
pointers into the source files, so no data is actually copied.

### What to expect

- 128 trajectories → 4,608 training samples; 5 epochs ≈ 25 min on a Quadro P4000.
- Validation loss: 0.156 → ~0.089 over 5 epochs.
- Test relative L1 error: **u ≈ 26%, c ≈ 7%** (overall ≈ 16.5%).
- Outputs: `checkpoints/wave_layer_finetuned.pt`, `results/wave_layer_predictions.png`,
  `results/wave_layer_error_distribution.png`.

### Key implementation details

- Two-channel input `[u, c]` (4-ch pretrained → 2-ch task) handled by
  `ignore_mismatched_sizes=True`, which re-initializes only the embedding and
  patch-recovery layers. These get a 10× higher learning rate.
- `channel_slice_list_normalized_loss=None` → plain L1 loss (avoids NaN issues).
- Time conditioning is ON: the model receives a normalized `time` scalar.

---

## 3. Tutorial 2 — Poisson-Gauss (Poisson Equation, Steady-State)

### Physics

The Poisson equation ∇²u = f, where the source f is a sum of Gaussian bumps.
This is the same mathematical structure as steady-state thermal conduction and
potential theory — the closest analogue to a static structural analysis in this
collection. The model maps the source field f directly to the solution field u.

### Run

```bash
python solve_poisson_gauss_tutorial.py
```

Single-file dataset (~2.6 GB) — no assembly step needed.

### What to expect

- 2,455 training samples (1/8 of the full 19,640 — see `NUM_TRAJECTORIES = -8`);
  5 epochs ≈ 10 min on a Quadro P4000.
- Validation loss: 0.068 → ~0.010 over 5 epochs.
- Test relative L1 error: **u ≈ 14%**.
- Outputs: `checkpoints/poisson_gauss_finetuned.pt`, `results/poisson_gauss_predictions.png`,
  `results/poisson_gauss_error_distribution.png`.

### Key implementation details

- **`use_conditioning=False`**: steady problems have no `time` input; the model is
  built without `ConditionalLayerNorm` (mirrors how scOT's own `train.py` handles
  `BaseDataset`-derived problems).
- One channel in (source f), one channel out (solution u).
- Visualization shows source → target → prediction → |error| side by side.

---

## 4. Tutorial 3 — SE-AF Airfoil (Steady Compressible Euler)

### Physics

Steady-state compressible Euler flow over **10,869 different airfoil geometries**.
The model receives the airfoil shape mask and predicts the steady density field ρ,
reproducing the suction-side low-density pocket and shock/expansion structures.
This is the classic "geometry in, field out" CAE surrogate workflow.

### Run

```bash
python solve_airfoil_tutorial.py
```

Single-file dataset (~1.4 GB).

### What to expect

- 1,313 training samples (1/8 of 10,509); 5 epochs ≈ 5 min on a Quadro P4000.
- Validation loss: 0.310 → ~0.114 over 5 epochs.
- Test relative L1 error: **ρ ≈ 17%** (errors concentrate at the shock line — the
  physically hardest, discontinuous feature).
- Outputs: `checkpoints/airfoil_finetuned.pt`, `results/airfoil_predictions.png`,
  `results/airfoil_error_distribution.png`.

### Key implementation details

- **`pixel_mask`**: pixels where the input equals 1 (airfoil body / masked region)
  carry a dummy label. The script passes `pixel_mask` to the model, which replaces
  the prediction at those pixels with the label — excluding them from the loss.
  The evaluation similarly masks them out before computing relative L1 error.
- Steady problem: `use_conditioning=False`, no `time` argument.


---

## 5. Customizing the Tutorials

All three scripts share the same structure, so the same knobs apply everywhere
(they are defined in the `# === Config ===` block at the top of each script):

| Setting | Default | Effect |
|---------|---------|--------|
| `NUM_EPOCHS` | 5 | Training epochs. Increase for better accuracy. |
| `BATCH_SIZE` | 8 | Reduce to 4 if you hit CUDA OOM on an 8 GB GPU. |
| `NUM_TRAJECTORIES` | -8 | Training-set size: `-1` = full, `-2` = half, `-8` = 1/8. |
| `LEARNING_RATE` | 5e-5 | Backbone learning rate. |
| `LR_EMBEDDING_RECOVERY` | 5e-4 | Learning rate for re-initialized embedding/recovery layers. |

To run with the **full training set** for publication-quality numbers, see
**Tutorial 8** (`solve_fulldata_tutorial.py`) — it sets `NUM_TRAJECTORIES = -1`,
adds LR warmup and early stopping, and explains the runtime cost (the original
paper fine-tunes for ~200 epochs with early stopping).

### Applying the pattern to other Poseidon datasets

The same script skeleton works for every downstream dataset in the
[Poseidon collection](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14).
To adapt one, change:

1. `DATASET_NAME` — e.g. `fluids.incompressible.Sines`, `fluids.compressible.Riemann`,
   `reaction_diffusion.AllenCahn` (see the mapping table in `README.md`).
2. `DATASET_REPO` / `DATA_FILE` — the matching HuggingFace dataset repository and file.
3. Time handling — time-dependent datasets (in `BaseTimeDataset`) need
   `use_conditioning=True` and a `time` argument in the forward call; steady datasets
   (`BaseDataset`) need neither.
4. Dataset URL — check the exact file names on the dataset's HuggingFace page
   (some are chunked and need assembly, like Wave-Layer; most others are single files).

---

## 6. Repository Structure

```
poseidon-tutorials/
├── guide.md                        ← this guide
├── TUTORIAL.md                     ← background notes (Wave-Layer focused)
├── README.md                       ← original Poseidon repository README
├── requirements.txt                ← Python dependencies
├── pyproject.toml                  ← scOT package config (pip install -e .)
│
├── solve_wave_pde_tutorial.py      ← Tutorial 1: acoustic wave (time-dependent)
├── solve_poisson_gauss_tutorial.py ← Tutorial 2: Poisson equation (steady)
├── solve_airfoil_tutorial.py       ← Tutorial 3: airfoil Euler (steady, pixel_mask)
├── solve_rollout_tutorial.py       ← Tutorial 4: autoregressive rollout (reuses Wave-Layer)
├── solve_helmholtz_tutorial.py     ← Tutorial 5: Helmholtz (steady, per-sample groups)
├── solve_rayleigh_taylor_tutorial.py ← Tutorial 6: Rayleigh-Taylor (gravity, 5-channel)
├── solve_richtmyer_meshkov_tutorial.py ← Tutorial 7: Richtmyer-Meshkov (shock-driven)
├── solve_fulldata_tutorial.py      ← Tutorial 8: full-data production run (reuses Poisson-Gauss)
├── solve_timewrapper_tutorial.py   ← Tutorial 9: TimeWrapper A/B experiment (reuses Poisson-Gauss)
│
├── scOT/                           ← Poseidon model source code (unmodified)
│   ├── model.py                    #    ScOT architecture + ScOTConfig
│   ├── problems/                   #    dataset loaders (get_dataset selector)
│   └── ...
├── configs/                        ← original W&B training configs (reference)
├── assets/                         ← paper figure
├── data/                           ← datasets are auto-downloaded here at runtime
│   ├── README.md                   #    dataset descriptions (kept in git)
│   └── assemble_data.py            #    original chunk-assembly script (kept in git)
├── results/                        ← output visualizations (sample results in git)
└── checkpoints/                    ← trained weights (NOT in git — re-train instead)
```

---

## 7. Advanced Tutorials (T4–T9)

Six additional self-contained scripts extend the three core tutorials with
non-overlapping learning goals. T4/T8/T9 need **zero new download** (they reuse
already-downloaded data); T5–T7 add ~19.3 GB of new datasets.

| # | Script | Dataset | New download | New skill |
|---|--------|---------|--------------|-----------|
| T4 | `solve_rollout_tutorial.py` | reuses Wave-Layer | **0 GB** | Autoregressive multi-step rollout, error-growth analysis |
| T5 | `solve_helmholtz_tutorial.py` | [Helmholtz](https://huggingface.co/datasets/camlab-ethz/Helmholtz) | 5.2 GB | Heterogeneous HDF5 layout (per-sample groups), BC input channel |
| T6 | `solve_rayleigh_taylor_tutorial.py` | [GCE-RT](https://huggingface.co/datasets/camlab-ethz/GCE-RT) | 5.4 GB | Small-data finetuning (1,260 trajs), gravity instability, fully dynamic 5-channel fields |
| T7 | `solve_richtmyer_meshkov_tutorial.py` | [CE-RM](https://huggingface.co/datasets/camlab-ethz/CE-RM) | 8.7 GB | Shock/discontinuous fields, dual-instability comparison |
| T8 | `solve_fulldata_tutorial.py` | reuses Poisson-Gauss | **0 GB** | Production recipe: full data (-1), warmup, early stopping |
| T9 | `solve_timewrapper_tutorial.py` | reuses Poisson-Gauss | **0 GB** | `.time` suffix / `TimeWrapper`: steady data through the conditioned path |

### T4 — Autoregressive Rollout (`wave.Layer.out`)

Uses the same Wave-Layer data with the `.out` suffix: instead of Tutorial 1's
single 0 → 14 jump, the model is finetuned on **all single-step transitions**
t → t+2 across the full horizon (10 transitions × dt=2 = 20 time units). The
script then performs a **true autoregressive rollout** — starting from t=0,
feeding each prediction back as input for 10 steps — and plots the key
production metric: the **error-growth curve** (teacher-forced vs
autoregressive). Zero new download.

### T5 — Helmholtz (`elliptic.Helmholtz`, steady)

Frequency-domain wave/scattering problem related to structural vibration.
Unlike the other datasets, the HDF5 file stores **per-sample groups**
(`Sample_i/a`, `Sample_i/bc`, `Sample_i/u`) instead of one big `solution`
array. The Dirichlet boundary value `bc` is broadcast into a **second input
channel**, conditioning the solution. 2-in/1-out channels, no time
conditioning, 512 test samples.

### T6 — Rayleigh-Taylor (`fluids.compressible.gravity.RayleighTaylor`)

Gravitationally-driven instability (bubble-and-spike mushroom growth) — the
only gravity dataset in the collection. Highlights: a **small-data regime**
(1,260 trajectories total vs 10,512 in Wave-Layer) showing how efficiently
the pretrained model adapts, and **fully dynamic 5-channel fields**
`[rho],[u,v],[p],[g]` where every channel evolves (unlike NS datasets where
rho/p stay constant, or Wave-Layer's constant c channel).

### T7 — Richtmyer-Meshkov (`fluids.compressible.RichtmyerMeshkov`)

Shock-driven counterpart of T6: an impulsive shock accelerates a density
interface, producing **discontinuous fields with sharp shock fronts** — the
hardest target for smooth-kernel neural surrogates. Same 4-channel structure
as GCE-RT, dt=2, horizon of 20 time units. Upstream quirk documented in the
script: the `.tracer` loader flag is accepted but ignored for this dataset.

### T8 — Full-Data Production Finetuning (reuses Poisson-Gauss)

The gap between the quick smoke-test runs of Tutorials 1–3 and paper-quality
numbers: `NUM_TRAJECTORIES = -1` unlocks the **full 19,640-trajectory training
set** (150x the quick runs), 20 epochs with cosine decay + **linear warmup**,
and **early stopping** (patience 5) that restores the best checkpoint. Tutorial
2's quick run reaches ~14% relative L1 error; this recipe lands clearly below
it. Zero new download — but budget several hours of GPU time instead of
minutes.

### T9 — TimeWrapper: Steady Data Through the Conditioned Path (reuses Poisson-Gauss)

A controlled experiment on identical data. scOT's `TimeWrapper` class (the
`.time` suffix) wraps any steady dataset so every sample carries `time = 1.0`,
letting it flow through the time-conditioned model path. The script finetunes
two models with the same budget — variant A: plain steady, no conditioning
(Tutorial 2 setup); variant B: `.time`-wrapped, `use_conditioning=True` — and
compares test errors in a bar chart. Use this decision procedure when routing
your own steady PDEs through mixed steady/transient pipelines. Zero new
download.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nan` training loss | Corrupted assembled dataset file (fill values ~1e36) | Delete `data/*.nc`, re-run — the script re-downloads |
| SSL certificate errors | Corporate proxy with SSL interception | Already handled automatically; check `curl` is on PATH |
| CUDA out of memory | 8 GB GPU with large batch | Set `BATCH_SIZE = 4` (or 2) |
| `cannot unpack non-iterable NoneType` (older copies) | Fixed bug in the Airfoil helper `forward_step` | Update the script to the current version |
| Slow download | HuggingFace throttling | Downloads are resumable — just re-run the script |
| `ImportError: scOT` | Editable install missing | Run `pip install -e .` from the repo root |

---

## 9. References

- **Paper**: [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101) (Herde et al., 2024)
- **Models**: [HuggingFace — Poseidon Collection](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)
- **Datasets**: [HuggingFace — Poseidon Downstream Tasks](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14)
- **License**: CC-BY-NC-4.0 (datasets and code — non-commercial use)

```bibtex
@misc{herde2024poseidon,
      title={Poseidon: Efficient Foundation Models for PDEs},
      author={Maximilian Herde and Bogdan Raoni{\'c} and Tobias Rohner and Roger K{\"a}ppeli
              and Roberto Molinaro and Emmanuel de B{\'e}zenac and Siddhartha Mishra},
      year={2024},
      eprint={2405.19101},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
