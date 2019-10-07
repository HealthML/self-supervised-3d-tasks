# Revisiting self-supervised visual representation learning

Tensorflow implementation of multiple self-supervised methods on 3D medical datasets.

## Overview

This codebase contains a re-implementation of four self-supervised representation learning
techniques, utility code for running training and evaluation loops (including on
TPUs) and an implementation of standard CNN models, such as ResNet v1, ResNet v2, U-Net (2D + 3D) and VGG19.

Specifically, we provide a re-implementation of the following self-supervised representation learning techniques:

1.  [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)
2.  [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)
3.  [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
4.  [Discriminative Unsupervised Feature Learning with Exemplar Convolutional
    Neural Networks](https://arxiv.org/abs/1406.6909)

## Usage instructions

In this codebase we provide configurations for training/evaluation of our models.

For debugging or running small experiments we support training and evaluation using a single GPU device.

### Preparing data

Please refer to the data_util package to see examples of producing the data in tf-records format.

### Clone the repository and install dependencies

Make sure you have [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Then perform the following commands, while you are in your desired workspace directory:
```
git clone https://gitlab.com/statistical-genomics/self-supervised-3d-tasks.git
cd self-supervised-3d-tasks
conda env create -f environment.yml
```

### Running locally on a single GPU

Run any experiment by running the corresponding shell script with the following
options, here exemplified for the fully supervised experiment:

```
./config/supervised/brats3d.sh 
```
Ensure that you have the correct desired flags in the shell script file.

You could start a tensorboard to visualize the training/evaluation progress:

```
tensorboard --port 2222 --logdir gs://<EVAL_DIRECTORY>
```

## Authors
Many parts of this code repository were reused from this [codebase](https://github.com/google/revisiting-self-supervised)