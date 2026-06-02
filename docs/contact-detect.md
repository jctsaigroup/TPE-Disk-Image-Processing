# Contact Detection

Contact detection is performed in two stages:

1. **Neighbour search:** For each frame, all pairs of particles within a distance threshold are considered as candidate contacts.

    The distance threshold is typically $r_i + r_j + d_{tol}$, where $r_i$ and $r_j$ are the radii of the two disks. $d_{tol}$ is set to 10 pixels for reasonable performance. Larger $d_{tol}$ would increase samples to classify and thus runtime, but smaller $d_{tol}$ might miss some contacts. 

    After locating all the ij pairs, the code handles the index:

    - Disks on the boundary are marked with `boundary=True`, and these disks can only be the `j` disk in a contact pair (i.e. they can only be contacted by interior disks, not contact other disks). Hence, there is also no contacts between two boundary disks.

    - Each $ij$ pair is only counted once to minimize the classification workload. 


2. **CNN classification:** Each candidate contact patch is cropped from the image:
<img src="../figures/contact_demo.png" style="width:50%; border-radius:4px"/>

    on the right is a contact, while on the left is a non-contact. These patches are then
    passed through a trained convolutional neural network (ResNet) to classify as contact or non-contact. The model returns a confidence score for each candidate, and only those above 0.5 are retained as detected contacts.

    The model is trained on similar experiment images of contact patches that are human-labelled. See [training](contact-detect-train.md) for details on the training pipeline and model architecture.


    After the detection, contacts between bulk disks are duplicated to form the reciprocal `ji` contact, which is convenient for later usage. Boundary contacts are not duplicated since they are directed by definition.

    Also, the force inversion process in step 3 cannot hangle disks with only one contact, since it cannot be in equilibrium. Hence, contact pairs that involve a disk with only one contact are marked as ``singular`` and can be filtered out when needed. This rarely happens, but is checked and handled for robustness.


## The Model

The classifier used in Step 2 (`02. TPE_contact_detect.ipynb`) is a binary ResNet18 model.

- Classes:
    - `0` = non-contact
    - `1` = contact
- Backbone: `torchvision.models.resnet18(weights=None)` at inference (weights loaded from trained checkpoint)
- Head:
    - `Linear(512 -> 1024)`
    - `ReLU`
    - `Dropout(0.5)`
    - `Linear(1024 -> 2)`

Typical checkpoint path in this pipeline:

- `models/ResNet18_contact_finetuned.pth`

Preprocessing used before inference (from `src.predict_contact_batch`):

- Crop contact-centered tangent patch for each candidate pair
- Resize to `128 x 128`
- Convert grayscale patch to RGB (3 channels)
- Scale to `[0, 1]`
- Apply ImageNet normalization

For each candidate pair, the model outputs 2 logits that are converted to softmax probabilities.
The pipeline stores:

- `contact = argmax(probabilities)`
- `prob = max(probabilities)`

Candidates predicted as class `1` are retained for downstream post-processing and force inversion.
 