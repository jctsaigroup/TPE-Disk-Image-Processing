# Fitting to Find Forces

This page explains the force-fitting stage implemented in `src/force.py`.
It covers synthetic image generation, loss terms, per-particle optimization, and ij/ji symmetrization.

## Main functions in code

- `synth_img_pytorch_residue(...)`: generate synthetic PE disk image from `(f, alpha, beta)`.
- `fit_disk_residue(...)`: optimize `f` and `alpha` for one disk.
- `fit_one_particle_cpu(...)` / `fit_one_particle_gpu(...)`: prepare one particle crop and run fitting.
- `symmetrize_forces(...)`: enforce reciprocal consistency using lower `fitLoss` side.

## 1. Synthetic image forward model

`synth_img_pytorch_residue` builds a disk grid and computes intensity with `StressSolve_residue_torch`.

Given force/angle vectors at all contacts on one particle:

- `f = [f_1, ..., f_z]`
- `alpha = [\alpha_1, ..., \alpha_z]`
- `beta = [\beta_1, ..., \beta_z]`

it produces a synthetic PE image for that disk. The grid/mask is cached in `_mesh_cache` for speed.

## 2. Optimization variables and constraints

`fit_disk_residue` optimizes `f0` and `alpha0` with Adam.

At each iteration:

1. Forces are projected positive by `f0_pos = abs(f0)`.
2. Synthetic image is generated from current variables.
3. Total loss is computed as:

$$
\mathcal{L} = \mathcal{L}_{img} + \mathcal{L}_{torque} + \mathcal{L}_{force}
$$

### Image term

$$
\mathcal{L}_{img} = \operatorname{mean}\left( G(I_{synth}) - I_{photo} \right)^2
$$

where $G(\cdot)$ is Gaussian blur (`smooth_image`, kernel size 3, sigma 1.0 in current code path).

### Torque equilibrium term

With weight $w_\tau = 10^5$ in code:

$$
\mathcal{L}_{torque} = w_\tau\left(\sum_k \sin(\alpha_k)\, r_m\, f_k\right)^2
$$

### Net force equilibrium term

Define `angle_term = alpha - beta + pi/2`.

$$
\mathcal{L}_{force} = \left(\sum_k \cos(\text{angle\_term}_k) f_k\right)^2 +
\left(\sum_k \sin(\text{angle\_term}_k) f_k\right)^2
$$

This penalizes non-zero resultant force components.

## 3. Convergence and stopping

`fit_disk_residue` stops when either:

- `n_iter` reached, or
- early stopping triggers: improvement below `tol` for `patience` consecutive checks, or
- loss becomes NaN/Inf.

Returned values:

- fitted force array
- fitted alpha array
- final scalar loss (`fitLoss`)
- loss history list

## 4. Per-particle wrappers

`fit_one_particle_cpu` and `fit_one_particle_gpu` do the same scientific steps:

1. Extract one particle crop using `get_disk_img`.
2. Convert to grayscale float image.
3. Read initial guesses from contact rows:
   - `force_pred`
   - `angle_pred`
   - `beta`
4. Run `fit_disk_residue`.
5. Write fitted values back into dataframe columns:
   - `force`
   - `alpha`
   - `fitLoss`

Particles with one or zero contacts are skipped (`z <= 1`).

## 5. ij/ji reciprocity correction

Each directed contact `(i, j)` and `(j, i)` is fit separately. They are then reconciled by `symmetrize_forces`.

### What `symmetrize_forces` requires

Input dataframe must contain:

- `frame`
- `i`, `j`
- `force`
- `alpha`
- `fitLoss`

### Selection rule

For each reciprocal pair in the same frame:

- compare `fitLoss(i,j)` and `fitLoss(j,i)`
- keep force/alpha from lower-loss side
- assign that same chosen value to both directions

It also returns diagnostic comparison metrics (`force_diff`, normalized force diff, wrapped alpha diff).

## 6. Practical tuning knobs

Most useful fitting knobs (from notebook config and `fit_disk_residue`):

- `lr`: Adam learning rate
- `n_iter`: max iterations
- `tol`, `patience`: early-stop sensitivity
- `fsigma`: photoelastic constant
- `plot_every`: debug visualization in sequential mode

Recommended workflow:

1. Validate on a small frame subset.
2. Inspect synthetic vs experimental image overlap.
3. Check reciprocity diagnostics after symmetrization.
4. Run full dataset once settings are stable.
