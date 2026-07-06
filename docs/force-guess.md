
# Contact Force Initial Guess

## Goal

For each detected contact ROI, the model predicts a force magnitude \(f\) (`force_pred`) and a contact angle \(\alpha\) (`angle_pred`):

<img src="figures/force_guess_patch.png" alt="Force guess" width="200"/>

These predictions are used as warm-start initial guesses for the [physics-constrained optimizer](https://linjunjr.github.io/TPE_Disk_Image_Processing/force-solve/), which then searches for the best solution. The guess isn't expected to be perfect — it just needs to be close enough to the true solution that the optimizer converges faster and avoids bad local minima.

## Why warm start matters

The inverse problem in [Step 3](https://linjunjr.github.io/TPE_Disk_Image_Processing/step3-force/) is non-convex, so starting from a random `(f, alpha)` is both slower and more likely to end in a poor local minimum — a good starting point matters. The original PeGS implementation uses \(G^2\) as an initial guess, but that fails under residual stress and gives no angle information at all. The ResNet guess is more robust: it usually lands the optimizer in a physically plausible basin from the start.

## The Model
 
The force-initialization network is a two-output ResNet18 regressor built in `src/force.py` via `get_model(device, output_dim=2)`.
 
| Component | Detail |
|---|---|
| Backbone | ResNet18, ImageNet-pretrained |
| Head | `Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→2)` |
| Output | `[force_pred, angle_pred]` |
| Training schedule | Backbone frozen during warmup, then unfrozen for fine-tuning |

## Training params



## Inference flow in Step 3

In `03. TPE_solve_force_vector.ipynb`, the model is built with `src.get_model(DEVICE)`, loaded from its trained `.pth` weights via `load_state_dict`, and switched to `model.eval()`. Contact ROIs are then extracted and preprocessed — resized to 224×224, replicated from grayscale to 3 channels, and normalized with ImageNet statistics — before being run through the model in batched forward passes. The results are written back to the bond table as `force_pred` and `angle_pred`.

These predictions are warm starts only; final values come from the physics-constrained fit described in [Fitting to Find Forces](https://linjunjr.github.io/TPE_Disk_Image_Processing/force-solve/).

