# Poseidon Wave-Layer PDE Finetuning Tutorial

This tutorial demonstrates how to **finetune a pretrained Poseidon-T foundation model**
on the **Wave-Layer acoustic wave equation PDE dataset** using a simple, self-contained
PyTorch training loop — **without W&B, without accelerate, without the scOT Trainer class**.

> **Note**: This is the original in-depth write-up for Tutorial 1. The repository now
> ships **nine self-contained tutorials** (rollout, Helmholtz, Rayleigh-Taylor,
> Richtmyer-Meshkov, full-data production run, TimeWrapper experiment, and more) —
> see **[`guide.md`](guide.md)** for the consolidated walkthrough and
> [`README.md`](README.md) for the tutorial index.

> **Prerequisite**: This tutorial builds on the [Poseidon](https://github.com/camlab-ethz/poseidon)
> framework. The original repository README is in [`README.md`](README.md).

---

## What You Will Learn

1. How to download and assemble the Wave-Layer dataset (~15 GB) using HDF5 Virtual Datasets
2. How to load a pretrained Poseidon-T model and adapt it to a new PDE (2-channel input)
3. How to build a parameter-group-aware optimizer (mirroring scOT's training strategy)
4. How to run a custom PyTorch training loop with cosine LR scheduling and gradient clipping
5. How to evaluate per-channel relative L1 error and visualize predictions

---

## Quick Start

### Step 1 — Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/camlab-ethz/poseidon.git
cd poseidon

# Create a virtual environment
python -m venv poseidon_env

# Activate it
# Windows:
poseidon_env\Scripts\activate
# Linux / macOS:
source poseidon_env/bin/activate

# Install PyTorch (choose your CUDA/CPU version from https://pytorch.org)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Install scOT as an editable package
pip install -e .
```

### Step 2 — Run the Tutorial

```bash
python solve_wave_pde_tutorial.py
```

The script runs end-to-end through 6 steps:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Download & assemble Wave-Layer dataset via HDF5 Virtual Datasets | `data/Wave-Layer.nc` (6 KB pointer file) |
| 2 | Load pretrained Poseidon-T, replace embedding/recovery for 2 channels | — |
| 3 | Build AdamW optimizer with 4 parameter groups (different learning rates) | — |
| 4 | Finetune with custom PyTorch loop (5 epochs, L1 loss, cosine LR) | `checkpoints/wave_layer_finetuned.pt` |
| 5 | Evaluate on 240 test samples (per-channel Relative L1 Error) | Console output |
| 6 | Visualize predictions vs. ground truth | `results/wave_layer_predictions.png` |

### Step 3 — Check Results

After training completes (~25 min on a single GPU):

```
checkpoints/
  └── wave_layer_finetuned.pt          # Model checkpoint (~83 MB)
results/
  ├── wave_layer_predictions.png        # Side-by-side: input / coeff / target / prediction / error
  └── wave_layer_error_distribution.png # Per-channel error histogram
```

---

## How It Works

### Data Pipeline

The Wave-Layer dataset contains solutions to the **acoustic wave equation** in a layered medium:

```
∂²u/∂t² = c(x,y) · ∇²u
```

- **`solution`**: shape `(10512, 21, 128, 128)` — wave field u(t, x, y) for 10,512 trajectories
- **`c`**: shape `(10512, 128, 128)` — spatially varying wave speed (layered medium, ~2235–3730 m/s)

The dataset is downloaded from [HuggingFace](https://huggingface.co/datasets/camlab-ethz/Wave-Layer)
as 4 chunked `.nc` files (~15 GB). The tutorial assembles them into a single logical file using
**HDF5 Virtual Datasets (VDS)** — a 6 KB file with pointers to the source files, avoiding the
need to copy 15 GB of data.

**Note on SSL/Proxy**: If you are behind a corporate proxy with SSL interception, the script
automatically:
- Sets `verify=False` on all `requests` calls (via monkey-patch)
- Uses `curl -k` (SSL bypass) for dataset downloads
- Disables `CURL_CA_BUNDLE` and `urllib3` SSL warnings

### Model Adaptation

Poseidon-T is pretrained on 4-channel data. The Wave-Layer task uses 2 channels (wave field `u`
and coefficient `c`). The script uses `ignore_mismatched_sizes=True` to:
- Replace the **embedding layer** (4ch → 2ch, randomly initialized)
- Replace the **patch recovery layer** (4ch → 2ch, randomly initialized)
- Keep all other pretrained weights (backbone, attention, etc.)

These replaced layers use a **10× higher learning rate** (5e-4 vs. 5e-5) to adapt faster.

### Training Strategy

The optimizer mirrors scOT's original Trainer with 4 parameter groups:

| Group | Parameters | Learning Rate | Weight Decay |
|-------|-----------|---------------|--------------|
| `standard` | Backbone weights (conv, attention) | 5e-5 | 1e-6 |
| `no_weight_decay` | LayerNorm, biases | 5e-5 | 0.0 |
| `embeddings` | Embedding + patch recovery (replaced) | 5e-4 | 1e-6 |
| `time_embedding` | ConditionalLayerNorm params | 5e-4 | 0.0 |

Training uses cosine LR scheduling, gradient clipping (max norm 5.0), and L1 loss.

### Expected Results

After 5 epochs of finetuning:

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.2781 | 0.1560 |
| 2 | 0.1355 | 0.1218 |
| 3 | 0.1037 | 0.1009 |
| 4 | 0.0837 | 0.0912 |
| 5 | — | 0.0890 (best) |

Test set performance (240 samples):

| Channel | Mean Relative L1 Error |
|---------|----------------------|
| u (wave field) | 25.89% |
| c (coefficient) | 7.15% |
| **Overall** | **16.52%** |

---

## File Structure

```
poseidon/
├── solve_wave_pde_tutorial.py    # ← Main tutorial script (run this!)
├── requirements.txt              # Python package dependencies
├── TUTORIAL.md                   # This file
├── README.md                     # Original Poseidon repo README
├── pyproject.toml                # scOT package config (for pip install -e .)
├── .gitignore                    # Git ignore rules
│
├── scOT/                         # Poseidon model source code
│   ├── model.py                  # ScOT architecture + ScOTConfig
│   ├── train.py                  # Original training script (uses accelerate/wandb)
│   ├── inference.py              # Original inference script
│   ├── metrics.py                # relative_lp_error
│   ├── utils.py                  # Utility functions
│   └── problems/
│       ├── base.py               # BaseTimeDataset, get_dataset()
│       └── wave/
│           └── acoustic.py       # Wave-Layer dataset loader
│
├── data/                         # Dataset directory (auto-downloaded)
│   ├── README.md                 # HuggingFace dataset description
│   ├── assemble_data.py          # Original assembly script
│   ├── solution_0.nc             # ~4.5 GB (gitignored)
│   ├── solution_1.nc             # ~4.5 GB (gitignored)
│   ├── solution_2.nc             # ~4.6 GB (gitignored)
│   ├── c_0.nc                    # ~657 MB (gitignored)
│   └── Wave-Layer.nc             # 6 KB VDS pointer (gitignored)
│
├── checkpoints/                  # Model checkpoints (gitignored)
│   └── wave_layer_finetuned.pt   # ~83 MB
│
└── results/                      # Output visualizations
    ├── wave_layer_predictions.png
    └── wave_layer_error_distribution.png
```

---

## Troubleshooting

### NaN Loss
If you see `nan` in the training loss, the assembled dataset file may be corrupted
(all fill values ~9.97e36). Delete `data/Wave-Layer.nc` and re-run — the script will
re-create the VDS file. Verify data integrity with:
```python
import h5py
with h5py.File('data/Wave-Layer.nc', 'r') as f:
    print(f['solution'][0, 0, :3, :3])  # should be ~1e-16, not ~1e36
```

### SSL Certificate Errors
If `curl` or `requests` fails with SSL errors behind a corporate proxy, ensure
the environment variables are set (handled automatically by the script):
```python
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
```

### CUDA Out of Memory
Reduce `BATCH_SIZE` in the script (default: 8). The model uses ~4 GB GPU memory at
batch size 8 with 128x128 grids.

---

## References

- **Paper**: [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101)
- **Models**: [HuggingFace — Poseidon Collection](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)
- **Dataset**: [HuggingFace — Wave-Layer](https://huggingface.co/datasets/camlab-ethz/Wave-Layer)
- **License**: CC-BY-NC-4.0 (dataset), see `LICENSE` for code

## Citation

```bibtex
@misc{herde2024poseidon,
      title={Poseidon: Efficient Foundation Models for PDEs},
      author={Maximilian Herde and Bogdan Raonic and Tobias Rohner and Roger Kappeli
              and Roberto Molinaro and Emmanuel de Bezenac and Siddhartha Mishra},
      year={2024},
      eprint={2405.19101},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
