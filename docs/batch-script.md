# Batch Script — `run_pipeline.py`

`run_pipeline.py` automates Steps 1 and 2 (disk tracking + contact detection) for one or more experiment directories from the command line.

!!! note
    Step 3 (force vector computation) is currently notebook-only and is not included in this batch script.

---

## Usage

### Single experiment

```powershell
python run_pipeline.py --img-dir "N:\PROJ_TPE\TPE_20260429A03_N=262x2_e-5rps"
```

### Multiple experiments (PowerShell loop)

```powershell
$dirs = @(
    "N:\PROJ_TPE\TPE_20260429A01_N=262x2_e-5rps",
    "N:\PROJ_TPE\TPE_20260429A03_N=262x2_2e-5rps"
)
foreach ($d in $dirs) {
    python run_pipeline.py --img-dir $d --verbose
}
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--img-dir` | *(required)* | Full path to the experiment image directory |
| `--pkl-dir` | `M:\Archive\Proj_TPE\Disk_traj_files` | Output directory for trajectory `.pkl` files |
| `--bond-dir` | `M:\Archive\Proj_TPE\Contact_bond_files` | Output directory for contact bond `.pkl` files |
| `--roi Y_MIN Y_MAX X_MIN X_MAX` | `250 1200 0 2000` | Region of interest in pixels |
| `--d-tol` | `10` | Neighbour distance tolerance (pixels) for contact search |
| `--verbose` | `False` | Save diagnostic figures alongside the output `.pkl` files |
| `--skip-tracking` | `False` | Skip Step 1; requires a trajectory `.pkl` to already exist |
| `--skip-contact` | `False` | Skip Step 2; run tracking only |

---

## Direct-run mode (no CLI flags)

For running directly from VS Code with "Run Python File", edit the constants near the top of `run_pipeline.py`:

```python
DIRECT_RUN_IMG_DIR = [
    r"N:\PROJ_TPE\TPE_20260509A01_N=262x2_e-5rps_2e-2fps_1000frames",
    r"N:\PROJ_TPE\TPE_20260510A_N=262x2_6SpeedSweep_REVERSE_strain=0.5_5e2FramesEach",
]

DIRECT_RUN_VERBOSE        = True
DIRECT_RUN_SKIP_TRACKING  = False  # set True to reuse existing .pkl
DIRECT_RUN_SKIP_CONTACT   = False
```

When the script is launched with no CLI arguments it reads these values instead of using `argparse`.

---

## Camera calibration

The script applies a homography correction to align the green and PE image coordinate systems. The calibration matrix is stored at the top of `run_pipeline.py`:

```python
CALIB_DATE = '2026-05-11'
DEFAULT_CALIB_H = np.array([
    [ 1.00897850e+00,  2.09385077e-02, -1.22728762e+00],
    [-1.80085815e-02,  1.01619232e+00, -4.11885450e+01],
    [-2.43424373e-06,  8.55305735e-07,  1.00000000e+00],
])
```

Replace the nine values when recalibrating the optical setup.
