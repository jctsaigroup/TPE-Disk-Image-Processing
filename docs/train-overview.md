# Training and Customizing Your Own Models

The models we provide in this repository are trained on a specific dataset and may not perform well on your data due to differences in imaging conditions, disk properties, or experimental setups. At some point you may want to train your own models to achieve better performance.

The training pipeline itself is in fact fairly standard stuff that is not specific to our project, and can be easily structured or explained by any AI agents. Fundamentals on machine learning can also be found in general machine learning resources such as the [official pytorch tutorial](https://pytorch.org/tutorials/). So this section will not explain many of the functions and classes that are fundamental building blocks of the training pipeline, such as the `Dataset` and `DataLoader` classes, the training loop, or the model architecture. We will not be providing a detailed documentation of how to use these either, as they are all well documented in their respective official documentations. 

Aside from that, as anyone who has worked with machine learning models can attest, model architectures and training hyperparameters are often empirically tuned to the dataset and task at hand, and the only way to optimize is to try it out. So we will also not be detailing the specific configs we chose for our models and why.  

Instead, what we will focus on is the part that is specific to this project: how to prepare your data and labels from the images and annotations you have. A minimal documentation of training configs and training workflow overview is still kept for reference.
