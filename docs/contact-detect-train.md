# Training Contact Detection (Step 2 Classifier)

This page documents how to generate training data and train the contact-detection CNN used in Step 2.
using [this repository](https://github.com/linjunJR/TPE_contact_detect_trainer).


# Data Generation Pipeline

This page documents the two-stage pipeline for generating and labeling training data for the CNN-based contact detector. The goal is to produce a classified image dataset of grain-grain contact regions, which is used to train a binary classifier distinguishing true contacts from near-contacts or non-contacts.

The two notebooks are:

- `Crop_contactROI_forTraining.ipynb` — crops candidate contact regions from photoelastic images given known grain centroid positions
- `Label_GUI_contact_detect.ipynb` — presents each cropped region through an interactive GUI for manual binary labeling

---

## Stage 1: Cropping Contact ROIs

**Notebook:** `Crop_contactROI_forTraining.ipynb`

### What it does

For each frame, the notebook loads a pre-computed bond file (`.pkl`) containing the centroid positions and radii of all grain pairs that are candidates for contact. For each candidate pair $(i, j)$, it:

1. Computes the contact point — the location on the surface of grain $i$ in the direction of grain $j$
2. Rotates the local image patch so the inter-grain axis is aligned vertically
3. Crops a square region of size $1.2 \times r_i$ centered on the contact point
4. Resizes to a fixed **128×128 px** output

The rotation step is important: it puts all crops in a canonical orientation, reducing the variance the classifier needs to learn.

### Inputs

| Variable | Description |
|---|---|
| `bond_file` | `.pkl` file with bond table — must contain columns `frame`, `i`, `j`, `xi`, `yi`, `xj`, `yj`, `ri` |
| `PE_dir` | Directory of photoelastic images (grayscale `.png`, named `Ib_<frame+1>.png`) |
| `output_dir` | Destination for cropped images |
| `sampled_frames` | List of frame indices to be processed |
| `roi` | Spatial crop applied to each full frame before processing, as `(y_min, y_max, x_min, x_max)` |

### Toggle switches

```python
save = 1   # write crops to disk
plot = 0   # show last crop inline (useful for a quick sanity check)
```

### Sampling

Since adjacent frames may have very similar contact configurations, Frames should be selected at large intervals to cover different contact states without excessive data volume. Only **50% of bonds per frame** are randomly sampled (seeded at 42) to reduce data load, adjust the sampling ratio as you see fit.


### Output

Sequentially numbered PNG files (`0_.png`, `1_.png`, ...) saved to `output_dir`. Each file is a 128×128 grayscale crop centered on a candidate contact point in canonical orientation.



---

## Stage 2: Manual Labeling via GUI

**Notebook:** `Label_GUI_contact_detect.ipynb`

### What it does

Presents each cropped image one at a time in a Jupyter widget interface. The user clicks a class button to assign a label, which triggers saving (with augmentation) and automatically advances to the next image.

### Classes

The classifier is **binary**:

| Label | Meaning |
|---|---|
| `0` | No contact — grains are near but not touching |
| `1` | Contact — a visible force chain or bright fringe at the contact point |

### Augmentation on save

Each labeled image is saved with **2 copies** into its class subfolder:

1. Original crop
2. Vertically flipped copy

### GUI controls

| Button | Action |
|---|---|
| `0` / `1` | Assign label, save augmented copies, advance to next image |
| `Skip` | Advance without saving (use for ambiguous or corrupted crops) |
| `Undo` | Delete the last saved set of files and return to the previous image |

The status bar shows progress as `N / total labeled`.

### Directory structure

```
contact_detect_data/
├── contact_roi/              ← unsorted crops from Stage 1
└── contact_roi_sorted_YYYYMM/
    ├── 0/                    ← labeled non-contacts
    └── 1/                    ← labeled contacts
```

The GUI reads from `contact_roi/` and writes into the sorted subfolders. Each labeled image is **deleted from the unsorted folder** upon saving, so the unsorted folder empties as you work through it. The file index in the sorted folders is tracked automatically — resuming a labeling session will continue from the correct index without overwriting existing files.

### Practical notes

- Use **Skip** liberally for ambiguous cases — a noisy label is worse than no label
- The **Undo** button only works within the current session; it cannot recover files from a previous run
- After labeling, check the class balance between `0/` and `1/` folders before training; if one class is underrepresented, consider labeling more samples or applying augmentation to balance the dataset.


## Stage 3: Model Training

**Notebook:** `contact_detect_training_resnet.ipynb`

A pretrained ResNet18 is fine-tuned on the labeled 128×128 contact crops to perform binary classification — contact (`1`) vs. no-contact (`0`).

---

## Configuration

Edit these paths before running:

```python
DATA_PATH            = r'sample_data'          # root of sorted label folders (0/ and 1/)
WARMUP_MODEL_PATH    = 'contact_detect_warmup.pth'
FINETUNED_MODEL_PATH = 'contact_detect_finetuned.pth'
BATCH_SIZE           = 32
```

`DATA_PATH` should point to the sorted output from Stage 2, structured as:

```
contact_roi_sorted_YYYYMM/
├── 0/    ← no-contact crops
└── 1/    ← contact crops
```

Multiple sorted directories can be combined by pointing `DATA_PATH` at a merged folder before training.

---

## Data Loading

The full dataset is loaded via `ImageFolder`, then **cached entirely in RAM** as a tensor before training begins. This avoids repeated reads from a network drive on every epoch, which would otherwise be the dominant bottleneck.

Images are normalised using ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`), consistent with the pretrained ResNet18 weights. The dataset is split **80/20** into train and validation sets (seed 42).

---

## Model Architecture

A pretrained ResNet18 is used as the backbone, with its original classification head replaced by:

```
Linear(512 → 1024) → ReLU → Dropout(0.5) → Linear(1024 → 2)
```

---

## Training Strategy

Training proceeds in two phases to avoid destabilizing the pretrained backbone early on.

| Phase | Backbone | Optimizer | Learning rate | Max epochs | Early stopping |
|---|---|---|---|---|---|
| 1 — Warmup | Frozen | Adam | 1 × 10⁻³ | 200 | patience = 20 |
| 2 — Fine-tuning | Unfrozen | Adam | 1 × 10⁻⁶ | 200 | patience = 20 |

**Loss function:** CrossEntropyLoss with **label smoothing = 0.1**, which softens the one-hot targets slightly and reduces overconfidence on the training set — useful given the 4× augmentation in the labeling stage may introduce near-duplicate samples.

**Early stopping** monitors validation loss with a patience of 20 epochs; the best checkpoint (lowest validation loss) is saved to disk at each phase independently.

---

## Outputs

| File | Contents |
|---|---|
| `contact_detect_warmup.pth` | Best model weights from Phase 1 |
| `contact_detect_finetuned.pth` | Best model weights from Phase 2 (use this for inference) |
| `logs/YYYYMMDD_HHMM/` | TensorBoard logs for Phase 1 |
| `logs/YYYYMMDD_HHMM_ft/` | TensorBoard logs for Phase 2 |

To inspect training curves live or post-hoc:

```bash
tensorboard --logdir logs/
```

The notebook also plots concatenated loss and accuracy curves for both phases on a shared epoch axis (Phase 1 uses solid lines, Phase 2 uses dashed), making it easy to see whether fine-tuning continued to improve over the warmup result or plateaued quickly.

---

## Practical Notes

- Check class balance between `0/` and `1/` before training — if contacts are significantly underrepresented, consider oversampling class `1` or using a weighted loss
- The RAM caching step prints the total memory footprint; ensure sufficient RAM before loading a very large labeled set
- `contact_detect_finetuned.pth` is the model to use for inference; it represents the best validation checkpoint from the full network fine-tuning phase

