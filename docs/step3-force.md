# Step 3 — Force Vector Computation

**Notebook:** [`03. TPE_solve_force_vector.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/03.%20TPE_solve_force_vector.ipynb)  
**Kernel:** `torch_env`

*(A CPU-only variant [`03. TPE_solve_force_vector_CPU.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/03.%20TPE_solve_force_vector_CPU.ipynb) is also available for machines without a compatible GPU.)*

This notebook computes the magnitude and direction of the contact force at each contact starting from a ResNet-based initial guess, and constrained optimization enforcing mechanical equilibrium.

---

## What the notebook does

### 1. Initial force guess (ResNet regression)
For each contact region, the force [`regression model`]() predicts an initial force magnitude and contact angle $\alpha$. This gives a warm start for the optimizer.

### 2. Optimization with equilibrium constraints
The force magnitudes and angles are refined by minimizing the pixel-wise difference between the synthetic and real PE images for each non-boundary particle. Also, net force and net torques on each grain are added as loss terms to guide the optimization towards physically consistent solutions. 

### 3. Symmetrization
Each contact $(i, j)$ is fit independently from particles $i$ and $j$. The `symmetrize_forces` function compares the two estimates and keeps the one with lower fit loss, enforcing Newton's third law.

---

## Outputs

Whats different from the previous two notebooks is that this step is extra time-consuming, so there are intermediate outputs designed in case of crashes or OOM errors. The initial prediction from the ResNet is saved as a `PREDICTION` pickle. During the optimization, results for each frame are saved in a temporary csv file once that frame is processed, and at the end of the notebook, these are aggregated into a final `FORCE` pickle file:

| Column | Description |
|---|---|
| `frame` | Frame index |
| `i`, `j` | Particle IDs |
| `force` | Contact force magnitude (N) |
| `alpha` | Half-contact-angle (radians) |
| `beta` | Contact normal angle (radians) |
| `fitLoss` | Residual image loss after optimization |

---

## Step-by-step instructions

If you follow standard dirctory conventions in the previous two steps, all you need to change is the `EXP_NAME` variable. Running the prediction cell would take roughly 1 hour for 2000 frames of 200 disks on a GPU.

After the prediction is done and saved as a preliminary pickle, go ahead and run the optimization procedure below. The cells look pretty intimidating but they are mostly just GPU configs and fallback safenets, so just run them as is. Again, start with one or few frames to make sure thing look reasonable. You can turn on the `plot_every` option in the config cell to visualize each grain and its fitted results, or you can view the full-field synthetic image of one frame by running one of the below cells. If fitting is failing and adjustment is needed, find more details in the "How it works" section.

Fitting would take about 40 seconds per frame on a GPU. After each frame is done, the results are immediately saved to a temporary csv file, so if the notebook crashes or runs out of memory, you can just restart it and it will pick up from the last saved frame. After all frames are done, the csv files are aggregated into a final pickle file. Optionally delete the temporary csv files or the prediction pickle file to save disk space.
