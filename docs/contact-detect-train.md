# Training Contact Detection (Step 2 Classifier)

This page documents how to train the contact-detection CNN used in Step 2.
The training workflow is based on:

- https://github.com/linjunJR/TPE_contact_detect_trainer

## Purpose

The model is a binary classifier that predicts whether a contact ROI contains a true disk-disk contact:

- class `0`: no contact
- class `1`: contact

In the main pipeline, this model is used after geometric candidate generation to remove false positives.

## Repository workflow

The trainer repo uses a three-stage data-to-model process:

1. **Crop ROIs**
   - Notebook: `Crop_contactROI_forTraining.ipynb`
   - Inputs: raw TPE frames plus bond/geometry metadata.
   - Output: contact-centered cropped images.

2. **Label ROIs**
   - Notebook: `Label_GUI_contact_detect.ipynb`
   - Interactive labeling into class folders (`0`/`1`).
   - Includes optional random-rotation augmentation while saving.

3. **Train ResNet18**
   - Notebook: `contact_detect_training_resnet.ipynb`
   - Two-phase transfer learning with warmup then fine-tuning.

## Model architecture

The trainer uses an ImageNet-pretrained ResNet18 backbone.

- Backbone: ResNet18 (pretrained)
- Input: `128 x 128` RGB, ImageNet-normalized
- Head:
  - `Linear(512 -> 1024)`
  - `ReLU`
  - `Dropout(0.5)`
  - `Linear(1024 -> 2)`

Loss function:

- `CrossEntropyLoss(label_smoothing=0.1)`

## Data split and batching

From the training notebook/README:

- Train/validation split: **80/20**
- Random seed for split: **42**
- Batch size: **32**

The notebook caches loaded images in RAM before splitting (TensorDataset), which speeds training but needs adequate memory.

## Two-phase training schedule

### Phase 1: Warmup

- Freeze backbone parameters.
- Train only the classification head.
- Optimizer: Adam on head parameters.
- Typical LR: `1e-3`.
- Up to 200 epochs with early stopping (patience 20 in notebook).
- Best checkpoint saved (validation-loss based).

### Phase 2: Fine-tuning

- Unfreeze full network.
- Continue from best warmup checkpoint.
- Lower LR for full-network optimization (notebook uses `1e-6`).
- Up to 200 epochs with early stopping.
- Save best fine-tuned checkpoint.

## Outputs

The final deliverable is a trained classifier checkpoint (`.pth`) that can be loaded by the contact-detection inference code in Step 2.

Typical repo output paths:

- warmup checkpoint (example): `contact_detect_warmup.pth`
- fine-tuned checkpoint (example): `contact_detect_finetuned.pth`

## Recommended training procedure for your pipeline

1. Build a balanced ROI dataset from your own experiments.
2. Verify labels carefully around near-touching disks (hard negatives).
3. Run warmup to convergence, then fine-tune.
4. Evaluate false positives/false negatives on held-out frames from unseen experiments.
5. Export the best checkpoint and update your Step 2 model path.

## Practical notes

- Keep preprocessing consistent between training and inference (ROI size, grayscale-to-RGB conversion, normalization).
- If recall is low, add more positive contacts and hard-negative examples.
- If precision is low, increase hard-negative coverage and review GUI labels for ambiguous boundary cases.

## References

- Trainer repository: https://github.com/linjunJR/TPE_contact_detect_trainer
- Main pipeline docs: `step2-contact.md`, `contact-detect.md`
