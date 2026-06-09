# Fitting to Find Forces

This page explains the force-fitting stage implemented in `src/force.py`.
It covers synthetic image generation, loss terms, per-particle optimization, and ij/ji symmetrization.
Below we first summarize the overall workflow and introduce the primary workhorse functions.

## Overview of workflow:

1. For each particle with 2 or more contacts, extract a cropped image and initial guesses for forces and angles.
2. From the initial guesses, generate a synthetic image with the residual stress added.
3. Compute the loss as a combination of image difference, torque imbalance, and net force imbalance.
4. Optimize the forces and angles with [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) until convergence or early stopping.
5. Iterate over all particles in all frames.
6. After all particles are fit, symmetrize forces $ij$ and $ji$.

## Primary workhorse functions

These functions are mostly optimized for GPU performance, so tensors are mostly torch objects, thats why many functions and variables have `_torch` suffix. The main functions are:

### 0. [StressSolve_residue_torch](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L145)

This function takes the contact forces and angles, computes the total stress ($\sigma = \sigma^\text{contact} + \sigma^\text{residual}$), then returns the intensity of a given pixel. It is called by `synth_img_pytorch_residue` to generate synthetic images for fitting.

### 1. [synth_img_pytorch_residue](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L175)

Calls `StressSolve_residue_torch` and iterates over all pixels of a disk crop. So basically, given force/angle vectors at all contacts on one particle:

- `f = [f_1, ..., f_z]`
- `alpha = [\alpha_1, ..., \alpha_z]`
- `beta = [\beta_1, ..., \beta_z]`

it produces a synthetic PE image for that disk.

This step is called in every iteration of the optimization loop.

### 2. [fit_disk_residue](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L225) 

The function optimizes `f0` and `alpha0` with the [Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

At each iteration:

1. Forces are projected positive by `f0_pos = abs(f0)`.
2. Synthetic image is generated from current variables using `synth_img_pytorch_residue`.
3. Total loss is computed as:

$$
\mathcal{L} = \mathcal{L}_{img} + \mathcal{L}_{torque} + \mathcal{L}_{force}
$$

#### Image term

This is the mean squared error between the synthetic and experimental image, after applying a Gaussian blur to both. The blur is important to make the loss landscape smoother and more convex, which helps optimization convergence.

$$
\mathcal{L}_{img} = \langle( I_{synth} - I_{experiment})^2 \rangle
$$


#### Force and Torque equilibrium term

This penalizes non-zero net force or torque. Here, an extra weight $w_\tau = 10^5$ is applied to the torque term, so that the three terms are roughly on the same order of magnitude at the beginning of optimization. The force term is unweighted since it is already on the same scale as the image term.

$$
\mathcal{L}_{force} = \left(\sum_z \vec{f_k} \right)^2 , \quad 
\mathcal{L}_{torque} = w_\tau\left(\sum_z \sin(\alpha_k)\, |\vec{r_k} \times \vec{f_k}|\right)^2
$$

#### Convergence and stopping

`fit_disk_residue` stops when either:

- `n_iter` is reached, or
- early stopping triggers: loss improvement below `tol` for `patience` consecutive iterations, or
- loss becomes NaN/Inf.

Returned values:

- fitted force array
- fitted alpha array
- final scalar loss (`fitLoss`)
- loss history list

### 4. [fit_one_particle_gpu](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L275)

Finally we wrap the whole per-particle fitting process in `fit_one_particle_gpu` and call it in the pipeline. The CPU version `fit_one_particle_cpu` does the same thing but with numpy arrays in parallel and without GPU acceleration
Both versions perform the same steps:

1. Extract one particle crop using `get_disk_img`.
2. Convert to grayscale float image.
3. Read initial guesses (from ResNet predictions) from contact rows:
4. Run `fit_disk_residue`.
5. Write fitted values `force`, `alpha`, `fitLoss` back into dataframe columns:

Particles with one or zero contacts are skipped (`z <= 1`).

## ij/ji reciprocity correction

Each directed contact `(i, j)` and `(j, i)` is fitted separately. They are then reconciled by `symmetrize_forces`.

### Selection rule

For each reciprocal pair in the same frame:

- compare `fitLoss(i,j)` and `fitLoss(j,i)`
- keep force/alpha from lower-loss side
- assign that same chosen value to both directions

It also returns diagnostic comparison metrics (`force_diff`, `alpha_diff`).


