# Train Your Own StarDist Model

This page summarizes the workflow used in:

- [manual_refine_GUI.ipynb](https://github.com/linjunJR/stardist-training-and-logistics/blob/main/manual_refine_GUI.ipynb)
- [mask_generator.ipynb](https://github.com/linjunJR/stardist-training-and-logistics/blob/main/mask_generator.ipynb)
- [TRAINING_stardist_disk_finder.ipynb](https://github.com/linjunJR/stardist-training-and-logistics/blob/main/TRAINING_stardist_disk_finder.ipynb)

The training process has three stages:

1. Manually refine disk centroids/radii in a tracking `.pkl`
2. Convert refined tracks into instance masks
3. Train and validate a `StarDist2D` model

---

## 1. Environment and folders

Use the `stardist_env` environment described in [Setup](setup.md).

Expected data layout (relative to `stardist training and logistics/`):

```text
stardist_data/
  TPE_disk/
    train/
      images/
      masks/
stardist_models/
```

The training notebook expects matching filenames in `images/` and `masks/`.

---

## 2. Refine trajectories manually (optional but recommended)

Notebook: `manual_refine_GUI.ipynb`

Purpose: correct missed or incorrect disk detections before generating masks.

### What to set

- `pickle_file`: input/output trajectory file
- `frame_files`: image sequence path (for example `Ic_*.png`)
- `roi`: crop region
- `camera_align(...)`: homography matrix for your current optical setup
- `frame_idx`: starting frame for editing

### Controls in the GUI

- Left click: add small disk
- Right click: add large disk
- `d`: delete mode, then click a disk to remove
- Left/Right arrow: move frames (step size is 5 in the notebook)
- `s`: save

The notebook writes the corrected dataframe back to the same `.pkl`.

---

## 3. Generate training masks from refined `.pkl`

Notebook: `mask_generator.ipynb`

Purpose: create paired image/mask files for StarDist instance segmentation.

### What to set

- `pickle_path`: refined trajectory file
- `circle_img_dir`: folder containing input circle/green-channel images
- `suffix`: filename suffix to prevent collisions across experiments
- `for frame in [...]`: frames to export
- `save` and `plot`:
  - first run with `save = 0`, `plot = 1` for visual check
  - then run with `save = 1` to write files

### Important preprocessing in this notebook

- `camera_align(...)` applies perspective correction
- `roi` crops the field of view
- `f['x'] = f['x'] - 100` trims the left boundary to match cropped images
- each disk gets a unique instance ID in the output mask (`uint16`)

Outputs are written to:

- `stardist_data/TPE_disk/train/images/*.tif`
- `stardist_data/TPE_disk/train/masks/*.tif`

Use identical filenames for each image-mask pair.

---

## 4. Train StarDist2D

Notebook: `TRAINING_stardist_disk_finder.ipynb`

### 4.1 Load and check data

The notebook:

- reads `*.tif` from `images/` and `masks/`
- asserts image and mask filenames match
- converts images to grayscale with `I[:,:,0]`
- normalizes each image with `normalize(x, 1, 99.8, axis=(0,1))`
- fills tiny label holes using `fill_label_holes`

Then it splits data with:

- random seed: `42`
- validation fraction: `15%` (minimum 1 image)

### 4.2 Augmentation used in your notebook

`augmenter(x, y)` applies:

- random flips along spatial axes
- intensity perturbation (`* U(0.9,1.1) + U(-0.5,0.5)`)
- Gaussian noise (`sigma = 0.02 * U(0,1)`)

If needed, disable augmentation by passing `augmenter=None` in `model.train`.

### 4.3 Model configuration

Current defaults in your notebook:

- `n_rays = 32`
- `grid = (4,4)`
- `n_channel_in = 1`
- `use_gpu = False and gputools_available()`

`grid = (4,4)` is intentionally used to enlarge effective field of view and improve performance for your dense disk scenes.

### 4.4 Save location and training call

Your notebook creates a timestamped run folder and trains for 300 epochs:

```python
from datetime import datetime
from stardist.models import StarDist2D

result_folder_name = datetime.now().strftime("%m-%d_%H-%M")
model = StarDist2D(conf, name=result_folder_name, basedir=r"C:\Users\jcTSAI\Desktop\stardist_models")
h = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=300)
```

Recommended adjustment: change `basedir` to your project folder, for example:

```python
model = StarDist2D(conf, name=result_folder_name, basedir="stardist_models")
```

---

## 5. Optimize thresholds

After training, run:

```python
model.optimize_thresholds(X_val, Y_val)
```

This selects:

- `prob_thresh`: confidence threshold for keeping a candidate instance
- `nms_thresh`: non-maximum suppression threshold for overlapping candidates

The optimized values are saved in `thresholds.json` inside the run folder.

---

## 6. Evaluate results

The notebook provides two checks:

1. Overlay predictions on validation images
2. Overlay predictions on a specific test frame (`test_image_filename`)

It also computes dataset metrics across IoU thresholds (`tau = 0.1 ... 0.9`):

- precision, recall, accuracy, f1
- mean scores and panoptic quality
- TP/FP/FN counts

Use these plots to compare runs and pick your final model.

---

## 7. Reuse trained model in the pipeline

Each trained run folder contains files such as:

- `weights_best.h5`
- `weights_last.h5`
- `thresholds.json`

Copy the selected model folder into your pipeline model location (for example `models/stardist_model/`) and point your disk-tracking step to that model.

---

## 8. Practical tips from this workflow

- Keep camera alignment and ROI consistent between mask generation and inference.
- Use filename suffixes (for example `_dense0919A01.tif`) when mixing data from multiple experiments.
- Start with quality over quantity: a smaller, clean, manually corrected set usually trains better than a large noisy set.
- If predictions merge neighbors too often, check masks first, then adjust `nms_thresh` and retrain if needed.
