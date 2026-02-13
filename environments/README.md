# Environment Setup

This project uses 3 different conda environments for different analysis stages.

## Environment Overview

| Environment | Notebook | Key Packages | Purpose |
|-------------|----------|--------------|---------|
| `env01_tracking` | 01_disk_tracking | stardist, tensorflow, trackpy | Detect and track photoelastic disks |
| `env02_contact` | 02_contact_detect | opencv, scikit-image | Detect particle contacts |
| `env03_force` | 03_force_solver | pytorch, torchvision | Force reconstruction with ResNet |

## Quick Installation

### Install all environments at once:
```bash
cd environments/
conda env create -f env_01_tracking.yml
conda env create -f env_02_contact.yml
conda env create -f env_03_force.yml
```

## Usage

### In Jupyter Notebook/Lab:
When opening each notebook, select the corresponding kernel:
- `01_TPE_disk_tracking_stardist.ipynb` → Kernel: **env01_tracking**
- `02_TPE_contact_detect.ipynb` → Kernel: **env02_contact**
- `03_TPE_solve_force_vector_with_ResNet_guess.ipynb` → Kernel: **env03_force**

### In VS Code:
1. Open a notebook
2. Click the kernel selector in the top right
3. Select the matching environment from the list

