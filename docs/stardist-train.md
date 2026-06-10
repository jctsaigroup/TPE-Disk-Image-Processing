# Train Your Own StarDist Model

This page summarizes the workflow used in the [TPE_stardist_disk_tracking_trainer](https://github.com/linjunJR/TPE_stardist_disk_tracking_trainer ) repo, which contains three notebooks:

1. [manual_refine_GUI.ipynb](https://github.com/linjunJR/TPE_stardist_disk_tracking_trainer/blob/main/manual_refine_GUI.ipynb)
2. [mask_generator.ipynb](https://github.com/linjunJR/TPE_stardist_disk_tracking_trainer/blob/main/mask_generator.ipynb)
3. [TRAINING_stardist_disk_finder.ipynb](https://github.com/linjunJR/TPE_stardist_disk_tracking_trainer/blob/main/TRAINING_stardist_disk_finder.ipynb)

The second and third notebooks are are largely adapted from the [StarDist2D training example](https://github.com/stardist/stardist/tree/main/examples/2D). 
If you already have nicely labelled data, you can skip the first notebook and directly prepare your paired image/mask files for training, and things should turn out well.
Otherwise, the first notebook is designed to help you refine your disk detections.

---

## Stage 1: Refine trajectories manually (optional but recommended)

Notebook: `manual_refine_GUI.ipynb`

Purpose: correct missed or incorrect disk detections before generating masks. If you already have perfect tracking results, you wouldn't need to train a model anyway. So presumably your trajectory file has some issues that require manual correction, and this notebook provides a convenient GUI to do that. The GUI allows you to use mouseclicks to add missing disks, delete false positives, and quickly iterate through frames.

You might notice the precision of the corrected disk centers are limited by the mouseclick-based input, but that's usually sufficient for training a model that can learn to predict more precise centers. The key is to provide good coverage of the different disk appearances and configurations in your data, so the model can learn the right features.

### What to set

- `pickle_file`: input/output trajectory file
- `frame_folder`: image sequence path  
- `camera_align(...)`: homography matrix to map your colored camera to the standard frame of reference.
- `frame_idx`: starting frame for editing
- `step`: step size for frame sampling in the GUI

### Controls in the GUI

- Left click: add small disk
- Right click: add large disk
- `d`: delete mode, then click a disk to remove
- Left/Right arrow: move frames (step size is defined by `step`)
- `s`: save

The notebook writes the corrected dataframe back to the same `.pkl`.
When you run the last cell with `save = 1`, it will overwrite the original file, so make a backup if you want to keep the unrefined version.


---

## Stage 2: Environment and folders

Use the `stardist_env` environment described in [environments](https://github.com/linjunJR/TPE_Disk_Image_Processing/tree/main/environments).

Expected data layout (relative to `stardist training and logistics/`):

```text
stardist_data/
      images/
      masks/
stardist_models/
```

The training notebook expects matching filenames in `images/` and `masks/`.


---

## Stage 3: Generate training masks from refined `.pkl`

Notebook: `mask_generator.ipynb`

Purpose: create paired image/mask files for StarDist instance segmentation.

### What to set

- `pickle_file`: refined trajectory file
- `frame_folder`: folder containing input circle/green-channel images
- `for frame in [...]`: frames to export
- `save` and `plot`:
  - first run with `save = 0`, `plot = 1` for visual check, then run with `save = 1` to write files

### Output files

Outputs are written to:

- `stardist_data/TPE_disk/train/images/*.tif`
- `stardist_data/TPE_disk/train/masks/*.tif`

The output files are set to have identical filenames for each image-mask pair.

---

## Stage 4: Train StarDist2D

Notebook: `TRAINING_stardist_disk_finder.ipynb`

Again, go through the [StarDist2D training example](https://github.com/stardist/stardist/tree/main/examples/2D), the FAQs and the original paper for fundamentals of how the model works and what the different parameters mean. Here we will just summarize the key points of how we adapted the training workflow for our specific data and goals. 

### Data Preparation

Training data is organized under `stardist_data/` with separate `images/` and `masks/` subdirectories. Images and masks are matched by filename (`.tif` format) and must have identical shapes — the notebook asserts this before proceeding.

Since the raw images are multi-channel, only the first channel is used:

```python
X = [I[:,:,0] for I in X]  # convert to grayscale
```

Images are normalized to the 1st–99.8th percentile intensity range, which robustly handles outlier bright pixels without clipping useful signal. Small holes in label masks are filled via `fill_label_holes` to avoid spurious interior boundaries.

A 15% validation split is applied via random permutation (seeded at 42 for reproducibility).

---

### Data Augmentation

A custom augmenter is applied during training to improve robustness on limited data. Three operations are applied per sample:

| Augmentation | Details |
|---|---|
| Random flip | Applied independently on each spatial axis with 50% probability |
| Intensity jitter | Multiplicative factor in [0.9, 1.1] plus additive offset in [−0.5, 0.5] |
| Gaussian noise | Standard deviation sampled uniformly from [0, 0.02] |

The intensity jitter range is intentionally conservative — enough to prevent overfitting to specific illumination conditions, but not so aggressive as to distort the signal in densely packed regions where contrast between neighboring disks matters.

To disable augmentation, pass `augmenter=None` to `model.train()`.

---

### Model Configuration

The model uses `Config2D` with the following settings:

```python
conf = Config2D(
    n_rays       = 32,
    grid         = (4, 4),
    use_gpu      = False,
    n_channel_in = 1,
)
```

**`n_rays = 32`** — the number of radial directions used to parameterize each object's boundary. 32 is the standard default and provides sufficient angular resolution for circular disks. Increasing this would add marginal shape detail at higher computational cost; decreasing it would degrade boundary precision.

**`grid = (4, 4)`** — predictions are made on a subsampled grid (one prediction every 4 pixels in each dimension) rather than at every pixel. This is the most important deviation from default settings for our data. With large disk diameters relative to image size, a full-resolution prediction grid is unnecessary and computationally wasteful. The 4×4 grid provides an adequate field of view while significantly speeding up both training and inference.

**`use_gpu = False`** — GPU-accelerated data generation via `gputools` is disabled. This is a deliberate choice for stability; training still runs on GPU if CUDA is available through PyTorch/TensorFlow.

**`n_channel_in = 1`** — single grayscale channel input, consistent with the preprocessing step above.

---

### Training

The model is trained for **300 epochs**:

```python
h = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=300)
```

The trained model is saved automatically to the desktop under a timestamped folder (`MM-DD_HH-MM`), which avoids overwriting previous runs during iterative experimentation.

---

### Threshold Optimization

After training, the two detection thresholds are optimized on the validation set:

```python
model.optimize_thresholds(X_val, Y_val)
```

- **`prob_thresh`** — minimum predicted probability for a candidate object to be accepted. Higher values reduce false positives at the cost of missing low-confidence detections.
- **`nms_thresh`** — IoU threshold for non-maximum suppression. Lower values suppress more overlapping candidates; for densely packed disks with near-zero gaps, this may need careful tuning to avoid merging adjacent particles.

The optimized values are stored with the model and used automatically at inference time.

---

### Evaluation

Detection performance is assessed at multiple IoU thresholds (0.1–0.9) on the validation set, reporting precision, recall, F1, mean matched score, and panoptic quality. These curves give a full picture of how conservative or permissive the model is across varying overlap tolerances.

Visual inspection is also supported — the notebook overlays predicted instance masks on grayscale images for both validation frames and specific test images, which is useful for catching systematic errors (e.g., split detections at disk boundaries or merged detections in high-density regions).

### Practical tips from this workflow

- Keep camera alignment and ROI consistent between mask generation and inference.
- Use filename suffixes (for example `_dense0919A01.tif`) when mixing data from multiple experiments.
- Start with quality over quantity: a smaller, clean, manually corrected set usually trains better than a large noisy set.
- If predictions merge neighbors too often, check masks first, then adjust `nms_thresh` and retrain if needed.
