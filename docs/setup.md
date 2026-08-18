# Setup & Installation

This pipeline currently contains **two separate conda environments** because TensorFlow 2.10 (needed by StarDist) and PyTorch 2.x use incompatible NumPy ABI versions. 

The pipeline works with or without a CUDA-capable GPU. If intended for GPU use, watch your CUDA versions when installing — the provided `environments/` YAML files are configured for NVIDIA GPUs with CUDA 11.2 (for TF) and CUDA 12.6 (for PyTorch). Versions may vary based on your machine and driver; check the [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) and [PyTorch CUDA support](https://pytorch.org/get-started/previous-versions/) pages for guidance.

Without a compatible GPU, the pipeline will run in CPU mode but expect roughly 1.5X longer runtimes for the first two steps. The third step offers a CPU friendly version that is optimized for speed using parallelization.

Check out the [StarDist Docs](https://github.com/stardist/stardist) and [PyTorch Docs](https://pytorch.org/) for more details on GPU requirements and troubleshooting.


| Environment | Pipeline Steps | GPU backend | Key packages |
|---|---|---|---|
| `stardist_env` | Step 1 — Disk tracking | TF 2.10 + CUDA 11.2 | TensorFlow-GPU, StarDist, CSBDeep, Trackpy |
| `torch_env` | Steps 2 & 3 — Contact detect & Force solve | PyTorch + CUDA 12.6 | PyTorch 2.6+cu126, Torchvision |

---

## Prerequisites

=== "stardist_env"

    - NVIDIA driver ≥ 450.80.02
    - CUDA 11.2 system libraries *(installed automatically via `cudatoolkit=11.2`)*

=== "torch_env"

    - NVIDIA driver ≥ 525.0 (for CUDA 12.6)
    - No system CUDA install needed — PyTorch CUDA libraries are bundled in the pip wheel

---

## Installation

```bash
cd environments/
conda env create -f stardist_env.yml
conda env create -f torch_env.yml
```

This will create both environments from the pinned `environments/` YAML files.

---

## Environment Selection

Activate the appropriate environment before running each pipeline step:

| Pipeline Script | Environment |
|---|---|
| `run_tracking.py` | `stardist_env` |
| `run_contact.py` | `torch_env` |
| `run_force.py` | `torch_env` |


### Notebook Demos (Optional)

If using the interactive notebooks in `Notebooks/`, select the matching kernel in VS Code / JupyterLab:

| Notebook | Kernel |
|---|---|
| `01. TPE_disk_tracking_stardist.ipynb` | `stardist_env` |
| `02. TPE_contact_detect.ipynb` | `torch_env` |
| `03. TPE_solve_force_vector.ipynb` | `torch_env` |

---

## Required Models

The pre-trained models must be placed in the `models/` folder before running:

```
models/
├── stardist_model/      ← StarDist2D model for disk segmentation
├── contact_model.pth    ← CNN contact classifier 
└── force_model.pth      ← ResNet force regressor 
```

