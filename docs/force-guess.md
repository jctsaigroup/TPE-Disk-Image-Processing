# Contact Force Initial Guess

This page documents the initial guess stage for contact force inversion.

## Goal

For each detected contact ROI, predict:

- force magnitude $f$ (`force_pred`)
- contact angle $\alpha$ (`angle_pred`)

<img src="\figures\force_guess_patch.png" alt="Force guess" width="200"/>

These predictions are used as warm-start initial guesses for the [physics-constrained optimizer](force-solve.md), which will seek the best solution. The guess is not expected to be perfect, but it should be close enough to the true solution to help the optimizer converge faster and avoid bad local minima. 

## Why warm start matters

The inverse problem in [Step 3](step3-force.md) is non-convex. Starting from random `(f, alpha)` is slower and more likely to end in poor local minima, so a good starting point is crucial. The original PeGS implementation uses $G^2$ as an initial guess, which fails in the case of residual stress and offers no angle information. The ResNet guess usually puts the optimizer close to a physically plausible basin.

## The Model

The force-initialization network is a two-output ResNet18 regressor built in `src/force.py` via `get_model(device, output_dim=2)`.

- Backbone: ImageNet-pretrained ResNet18
- Head:
	- `Linear(512 -> 256)`
	- `ReLU`
	- `Dropout(0.2)`
	- `Linear(256 -> 2)`
- Output interpretation: `[force_pred, angle_pred]`

During training, all backbone layers are frozen in warmup and then unfrozen for fine-tuning.

## Inference flow in Step 3

In `03. TPE_solve_force_vector.ipynb`, the model is used as follows:

1. Build model with `src.get_model(DEVICE)`.
2. Load trained weights (`.pth`) with `load_state_dict`.
3. Switch to inference mode with `model.eval()`.
4. Extract and preprocess contact ROIs.
5. Run batched forward passes.
6. Write results to bond table columns:
	 - `force_pred`
	 - `angle_pred`

Typical preprocessing before inference:

- Resize ROI to `224 x 224`
- Replicate grayscale to 3 channels
- Apply ImageNet normalization

These predictions are warm starts only; final values are obtained by the physics-constrained fit described in [force-solve.md](force-solve.md).


