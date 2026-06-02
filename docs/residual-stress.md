# Simulating Residual Stresses

This page explains the residual-stress term used in Step 3 force inversion.
The implementation is in [StressSolve_residue_torch](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L145) in [force.py](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py).

## What for?

The residual-stress is the non-contact-induced stress patterns inside the disks, presumably due to the fluids, that manifests as a bright rim around every disk. Since the [force inversion](force-solve.md) relies on fitting the contact-induced fringes to the observed image, the rim results in very bad precision when using traditional force fitting algorithm that considers a disk to start out totally stress-free. 

Note that this is a systematic bias on the **stress** level, not just the intensity, so it cannot be easily removed by image shifting pixel values. We must simulate a artificial stress profile, superpose it with the contact-induced stress, and let the optimizer fit the contact forces based off that.

## The stress profile

We construct an empirical form for the residual stress subject to the following constraints:

  1. Axisymmetric around the disk center, $\sigma_{r\theta} = \sigma_{\theta r} = 0$, and $\sigma_{rr}$ and $\sigma_{\theta\theta}$ are functions of $r$ only. The equilibrium condition then reads: $\frac{d\sigma_{rr}}{dr} + \frac{\sigma_{rr} - \sigma_{\theta\theta}}{r} = 0$.

  2. Free boundary: $\sigma_{rr}(R) = 0$ at the disk edge $R$. 

  3. $\sigma_{rr} - \sigma_{\theta\theta}$ maximize at $r = R$, resulting in the bright rim at the disk edge.

We formulate a simple polynomial form for that satisfies the above:

$$
\sigma_r^{res} = K_{res}(R^p - r^p)
$$

$$
\sigma_\theta^{res} = K_{res}(R^p - (p+1)r^p)
$$

$K_{res}$ is an overall scale factor that controls the intensity of the residual stress, and $p$ is a power that controls the radial profile shape. We observe that the rim is roughly at peak intensity, so we approximate the value of $K_{res}$ by setting the disk edge to be at the first fringe peak, which corresponds to
setting the maximum principal-stress difference at $r = R$ to be $F_\sigma/2$ (half fringe order):

$$
K_{res} = -\frac{F_\sigma}{\pi\,p\,R^p}\,\arcsin(1)
= -\frac{F_\sigma}{2\,p\,R^p}
$$

Here $F_\sigma$ is the photoelastic calibration scale that converts stress to intensity. 

$p$ is a free parameter that controls the shape of the residual stress. We find that $p=6$ gives a good fit to the observed rim profile, and we use that for all our simulations. Below is a simulated disk with residual  stress $p = 6$ that is subject to various diametric loads. Note the black cracks that opens up when forces are small nicely resembles the experimental images, and the width bright rim is also well captured.

<img src="\figures\res_stress_synth.png" alt="Residual stress profile" width="800"/>


## Where it appears in code

- [src.force.StressSolve_residue_torch(...)](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L145): finds the total stress from contact forces plus residual stress, then maps stress to intensity.
- [src.force.synth_img_pytorch_residue(...)](https://github.com/linjunJR/TPE_Disk_Image_Processing/blob/main/src/force.py#L225): calls the stress solver with `power=6` and generates a synthetic disk image on a `px x px` grid.



