# Step 2 — Contact Detection

**Notebook:** [`02. TPE_contact_detect.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/02.%20TPE_contact_detect.ipynb)  
**Kernel:** `torch_env`

This notebook identifies which particle pairs are in physical contact and classifies each candidate pair using a trained convolutional neural network (CNN).

---

## Inputs

| Variable | Description |
|---|---|
| `PKL_FILE` | Path to the trajectory `.pkl` from Step 1 |
| `IMG_DIR` | Directory containing PE images (`bw_*.png`) |
| `D_TOL` | Distance tolerance in pixels for neighbour search (default: `10`) |
| Contact model | Pre-trained CNN in `models/contact_model.pth` |

---

## What the notebook does

### 1. Neighbour search
For every frame, all pairs of particles whose centre-to-centre distance falls within $r_i + r_j + D\_TOL$ pixels are enumerated as *candidate contacts*.

### 2. Contact patch extraction
For each candidate pair $(i, j)$, a small image patch centred on the contact point is cropped from the PE image. The patch is rotated so the contact normal is always horizontal (canonical orientation).

### 3. CNN classification
The patch is passed through the pre-trained contact detection network (ResNet backbone). The network outputs a binary label:

- `1` → confirmed contact
- `0` → no contact (gap or near-miss)

along with a continuous confidence score.

### 4. Post-processing
Low-confidence detections can be filtered by a threshold. The resulting contact list is deduplicated so each $(i, j)$ pair appears once per frame.

---

## Outputs

A pickle file is saved containing a `pandas.DataFrame` with one row per *(contact, frame)*:

| Column | Description |
|---|---|
| `frame` | Frame index |
| `i`, `j` | Particle IDs forming the contact |
| `xi`, `yi` | Position of particle $i$ (pixels) |
| `xj`, `yj` | Position of particle $j$ (pixels) |
| `beta` | Contact angle (radians, measured from the vertical) |
| `score` | CNN classification confidence |

---

## Step-by-step instructions

If step 1 is done correctly, theres not much that can go wrong here. The CNN model do all the work here. Just make sure to set the correct paths at the top of the notebook:

```python
EXP_FOLDER = "TPE_20260518A03_N=262x2_step_shear_relax_e-3rpsx10s_stop100s_1fps_10reps"
TRAJ_DIR  = r'M:\Archive\Proj_TPE\Disk_traj_files'   # trajectory .pkl files
IMG_DIR   = r'N:\PROJ_TPE'                            # raw image root
BOND_DIR  = r'M:\Archive\Proj_TPE\Contact_bond_files' # output directory

```

Regardless, it is always good practice to first check by running only a few frames to check that nothing has gone wrong.