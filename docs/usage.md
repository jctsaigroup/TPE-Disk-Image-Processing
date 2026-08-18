# Usage Guide

The TPE Disk Image Processing pipeline is driven by YAML configuration files and command-line scripts. This page covers the standard workflow for processing experiments.

---

## Configuration Files

All pipeline parameters are controlled via two YAML files in the repository root:

### `dynamic_config.yaml`

Contains experiment-specific and frequently-changed settings:

```yaml
paths:
  data_dir: "N:/PROJ_TPE"              # Root directory containing experiment folders
  pkl_dir: "M:/Archive/Proj_TPE/Disk_traj_files"
  bond_dir: "M:/Archive/Proj_TPE/Contact_bond_files"
  force_dir: "M:/Archive/Proj_TPE/Force Inversion"
  calibration_file: "calibration_matrix/20260807.npy"

experiment:
  exp_folder: "TPE_20260807A01_N=262x2_stop_go_1e-3rpsx200s_stop1000s_1000framesx1fps_20reps"
  # Can also be a list for batch processing:
  # exp_folder: ["TPE_20260808A01_...", "TPE_20260808A02_...", "TPE_20260808A03_..."]

tracking:
  frame_selection:
    mode: "all"                 # "all", "random", "first", or "single"
    value: 1                    # depends on mode
  verbose: 1                    # Visualize diagnostic figures
  save: 1                       # Save output to pickle

contact:
  frame_selection:
    mode: "all"
    value: 1
  verbose: 1
  save: 1

force:
  gpu_parallel_workers: 4       # Number of particles to fit concurrently per frame
  plot_every: 0                 # Visualize each disk's fit (only works with gpu_parallel_workers=1)
  verbose: 0                    # Print fitting loss details
  frame_selection:
    mode: "all"
    value: 1
```

### `stationary_config.yml`

Contains static parameters that rarely change (model paths, ROI, thresholds, hyperparameters). See the file in the repository for the full structure.

---

## Running the Pipeline

The pipeline consists of three sequential steps. Each step must complete before running the next.

### Step 1: Disk Tracking

Detects and tracks disks using StarDist, computes orientation angles, and calculates per-particle G².

**Environment:** `stardist_env`

```bash
conda activate stardist_env
python run_tracking.py --config dynamic_config.yaml
```

**Key CLI options:**
```bash
--exp-folder TPE_20260808A01_...     # Override experiment folder
--frame-mode all|random|first|single # Override frame selection mode
--frame-value N                       # Number of frames (depends on mode)
--dry-run                             # Validate config without running
```

**Output:** `{pkl_dir}/{exp_folder}_traj.pkl`

---

### Step 2: Contact Detection

Identifies contacts between particles using a trained CNN classifier.

**Environment:** `torch_env`

```bash
conda activate torch_env
python run_contact.py --config dynamic_config.yaml
```

**Key CLI options:**
```bash
--exp-folder TPE_20260808A01_...     # Override experiment folder
--dry-run                             # Validate config without running
```

**Input:** Trajectory `.pkl` from Step 1  
**Output:** `{bond_dir}/{exp_folder}_bond.pkl`

---

### Step 3: Force Vector Computation

Computes contact force magnitudes and angles using ResNet initial guess + physics-based optimization.

**Environment:** `torch_env`

```bash
python run_force.py --config dynamic_config.yaml
```

**Key CLI options:**
```bash
--exp-folder TPE_20260808A01_...               # Override experiment folder
--force-frame-mode all|random|first|single     # Override frame selection
--force-frame-value N                           # Number of frames
--gpu-workers N                                 # Override GPU parallelism
--dry-run                                       # Validate config
```

**Example (process single frame 1401 with 8 workers):**
```bash
python run_force.py --config dynamic_config.yaml \
    --exp-folder TPE_20260521A01_... \
    --force-frame-mode single \
    --force-frame-value 1401 \
    --gpu-workers 8
```

**Inputs:** Trajectory `.pkl` + Contact bond `.pkl`  
**Output:** `{force_dir}/{exp_folder}_force.pkl`

---

## Batch Processing

Process multiple experiments in sequence by looping over experiment folders:

=== "PowerShell"

    ```powershell
    $experiments = @(
        "TPE_20260808A01_N=262x2_e-5rps",
        "TPE_20260808A02_N=262x2_2e-5rps",
        "TPE_20260808A03_N=262x2_5e-5rps"
    )

    foreach ($exp in $experiments) {
        conda activate stardist_env
        python run_tracking.py --config dynamic_config.yaml --exp-folder $exp
        
        conda activate torch_env
        python run_contact.py --config dynamic_config.yaml --exp-folder $exp
        python run_force.py --config dynamic_config.yaml --exp-folder $exp
    }
    ```

=== "Bash"

    ```bash
    for exp in TPE_20260808A01_... TPE_20260808A02_... TPE_20260808A03_...; do
        conda activate stardist_env
        python run_tracking.py --config dynamic_config.yaml --exp-folder "$exp"
        
        conda activate torch_env
        python run_contact.py --config dynamic_config.yaml --exp-folder "$exp"
        python run_force.py --config dynamic_config.yaml --exp-folder "$exp"
    done
    ```

Alternatively, you can specify a list of experiment folders directly in `dynamic_config.yaml`:

```yaml
experiment:
  exp_folder:
    - "TPE_20260808A01_N=262x2_e-5rps"
    - "TPE_20260808A02_N=262x2_2e-5rps"
    - "TPE_20260808A03_N=262x2_5e-5rps"
```

Then run normally:

```bash
python run_tracking.py --config dynamic_config.yaml
python run_contact.py --config dynamic_config.yaml
python run_force.py --config dynamic_config.yaml
```

The scripts will automatically loop through all experiments.

---

## Frame Selection Modes

All three scripts support flexible frame selection via `frame_selection.mode`:

| Mode | Description | `value` parameter |
|------|-------------|-------------------|
| `all` | Process every frame in the experiment | *(ignored)* |
| `random` | Process N randomly selected frames | N = number of random frames |
| `first` | Process the first N frames | N = number of frames |
| `single` | Process one specific frame | N = frame number |

**Examples:**

```bash
# Process all frames (default)
python run_tracking.py --config dynamic_config.yaml --frame-mode all

# Process 10 random frames
python run_tracking.py --config dynamic_config.yaml --frame-mode random --frame-value 10

# Process first 50 frames
python run_tracking.py --config dynamic_config.yaml --frame-mode first --frame-value 50

# Process only frame 1401
python run_force.py --config dynamic_config.yaml --force-frame-mode single --force-frame-value 1401
```

---

## Diagnostic Output

Set `verbose: 1` in the config or use visualization options to generate diagnostic figures:

- **Tracking:** Random frame with detected disks, angles, and G² values overlaid
- **Contact:** Sample frame with contact bonds drawn
- **Force:** Per-particle synthetic vs. experimental PE images (set `force.plot_every: 1` with `gpu_parallel_workers: 1`)

Figures are saved alongside output pickle files when `verbose` is enabled.

---

## Notebook Demos (Optional)

Interactive Jupyter notebooks are available in the `Notebooks/` folder for testing and exploration:

- `01. TPE_disk_tracking_stardist.ipynb`
- `02. TPE_contact_detect.ipynb`
- `03. TPE_solve_force_vector.ipynb`

These demonstrate the same processing steps interactively but require manual parameter editing and are not recommended for batch processing. Use the CLI scripts for production workflows.
