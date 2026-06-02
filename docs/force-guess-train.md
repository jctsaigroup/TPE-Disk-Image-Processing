# Training Contact Force Initial Guess (CNN Regressor)

This page documents training for the force-initialization CNN used before inverse fitting in Step 3.
The workflow is based on:

- https://github.com/linjunJR/TPE_contact_force_CNN_trainer

## Purpose

For each contact ROI, the regressor predicts:

- force magnitude
- force angle

These predictions are used as warm-start values (`force_pred`, `angle_pred`) before physics-constrained optimization in the force solver.

## Data generation pipeline

Because labeled real data is limited, the repo trains primarily from synthetic data with known ground-truth force vectors.

Primary notebooks:

1. `Contact_force_regression_label_produce.ipynb`
   - Rotates each disk image so the target contact is aligned.
   - Crops one contact-centered patch per contact.
   - Saves numbered images plus a `labels.npy` file.

2. `Contact_force_regression_TRAINING.ipynb`
   - Loads cropped images and `[force, angle]` labels.
   - Trains a two-output ResNet18 regressor (warmup + fine-tune).

Expected dataset format:

```text
<output_dir>/
    00001.png ... NNNNN.png
    labels.npy      # shape (N, 2): [force_magnitude, force_angle]
```

## Model architecture

The model in the training notebook matches your pipeline-side definition (`src/force.py`):

- Backbone: ImageNet-pretrained ResNet18
- Head:
  - `Linear(512 -> 256)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(256 -> 2)`

Input transform:

- resize to `224 x 224`
- grayscale to 3 channels
- ImageNet normalization

Output:

- two regression targets: `[force_magnitude, force_angle]`

Loss decomposition:

- `MSE(force) + MSE(angle)`

## Train/val/test split

From training notebook defaults:

- train: **70%**
- validation: **15%**
- test: **15%**

The notebook reports separate force and angle errors during train and validation, in addition to total loss.

## Two-phase training schedule

### Phase 1: Warmup (frozen backbone)

- Freeze all ResNet backbone layers.
- Train only regression head.
- Optimizer: Adam.
- LR: `1e-4`.
- Early stopping monitors validation loss.

Typical defaults from notebook:

- epochs: 100
- patience: 20
- min_delta: 0.001

### Phase 2: Fine-tuning (full network)

- Reload best warmup checkpoint.
- Unfreeze all layers.
- Continue training with lower LR.
- Optimizer: Adam.
- LR: `1e-5`.
- Early stopping on validation loss.

Typical defaults from notebook:

- epochs: 200
- patience: 20
- min_delta: 0.0001

## Evaluation

The trainer evaluates on held-out test data with:

- force true-vs-pred scatter (typically log-log)
- angle true-vs-pred scatter
- loss curves across warmup and fine-tuning

The final checkpoint is then copied into your pipeline model folder (for example under `models/`) and loaded by Step 3.

## How this connects to your Step 3 notebook

In `03. TPE_solve_force_vector.ipynb`, the loaded model predicts `force_pred` and `angle_pred` for each contact ROI. Those values are only the initial guess; final forces are obtained by per-particle optimization and reciprocity symmetrization.

## Practical recommendations

1. Keep the exact same preprocessing in training and inference.
2. Cover wide force ranges in synthetic data to avoid bias in low/high force tails.
3. Include realistic image noise/contrast variation in synthetic generation if transfer to experiment is weak.
4. Validate model quality by downstream fitting behavior, not only standalone MSE.

## References

- Trainer repository: https://github.com/linjunJR/TPE_contact_force_CNN_trainer
- Force initial guess overview: `force-guess.md`
- Force fitting stage: `force-solve.md`
