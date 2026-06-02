# TPE Disk Image Processing

This pipeline automates the full analysis workflow of a 2D granular experiment using **photoelastic disks** — from raw experimental images to a complete force network. It combines deep learning (StarDist, ResNet) with physics-based optimization under mechanical equilibrium constraints.

The approach is largely inspired by the established [PeGS algorithms](https://github.com/photoelasticity/PeGS2). But PeGS largely relies on a nice monotonic $G^2$ calibration curve. Wet photoelastic disks inevitably develop residual stress, which renders the classical $G^2$ method unusable. The custom-trained neural networks here are designed to work around this limitation.


<div style="display:flex; gap:16px; flex-wrap:wrap;">
  <div style="flex:1; min-width:200px">
    <img src="figures/original.png" alt="Experimental image" style="width:100%; border-radius:6px"/>
    <p style="text-align:center"><em>Raw experimental image</em></p>
  </div>
  <div style="flex:1; min-width:200px">
    <img src="figures/reconstructed.png" alt="Reconstructed image" style="width:100%; border-radius:6px"/>
    <p style="text-align:center"><em>Reconstructed from extracted positions & forces</em></p>
  </div>
</div>

---

## Workflow Overview

The pipeline consists of three main steps, each implemented as a Jupyter notebook. Each step results in a pickle file `.pkl` that stores a Pandas dataframe. The outputs of one step are the inputs to the next.

```mermaid
flowchart TD
    I1[Green Images] --> B[01. TPE_disk_tracking_stardist.ipynb]
    I2[UV Image] --> B
    I3[PE Image] --> B
    B --> C[Trajectory .pkl\n positions · angles · IDs · G²]
    C --> D[02. TPE_contact_detect.ipynb]
    D --> E[Contact Bond .pkl\n pairs · positions · angles]
    E --> F[03. TPE_solve_force_vector.ipynb]
    F --> G[Force .pkl\n magnitudes & angles of contact forces]

    style I1 stroke:#4CAF50,stroke-width:3px
    style I2 stroke:#2196F3,stroke-width:3px
    style I3 stroke:#FFC107,stroke-width:3px
    style C stroke:#FFA726,stroke-width:2px
    style E stroke:#FFA726,stroke-width:2px
    style G stroke:#FFA726,stroke-width:2px
    style B stroke:#9C27B0,stroke-width:2px
    style D stroke:#9C27B0,stroke-width:2px
    style F stroke:#9C27B0,stroke-width:2px
```

| Step | Notebook | Environment | Output |
|------|----------|-------------|--------|
| 1 | [`01. TPE_disk_tracking_stardist.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/01.%20TPE_disk_tracking_stardist.ipynb) | `stardist_env` | Trajectory `.pkl` |
| 2 | [`02. TPE_contact_detect.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/02.%20TPE_contact_detect.ipynb) | `torch_env` | Contact bond `.pkl` |
| 3 | [`03. TPE_solve_force_vector.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/03.%20TPE_solve_force_vector.ipynb) | `torch_env` | Force `.pkl` |
| 3 (CPU) | [`03. TPE_solve_force_vector_CPU.ipynb`](https://nbviewer.org/github/jctsaigroup/TPE-Disk-Image-Processing/blob/main/03.%20TPE_solve_force_vector_CPU.ipynb) | `torch_env` | Force `.pkl` (no GPU required) |

For batch/automated runs across many experiments, see the [Batch Script](batch-script.md) page.
