# Revisiting self-supervised visual representation learning

Keras implementation of multiple self-supervised methods on 2D and 3D medical datasets.

## Overview

This codebase contains a re-implementation of five self-supervised representation learning
techniques, utility code for running training and evaluation loops.

Specifically, we provide a re-implementation of the following self-supervised representation learning techniques:

1.  [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)
2.  [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)
3.  [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
4.  [Discriminative Unsupervised Feature Learning with Exemplar Convolutional
    Neural Networks](https://arxiv.org/abs/1406.6909)
5.  [Representation Learning withContrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)

## Usage instructions

In this codebase we provide configurations for training/evaluation of our models.

For debugging or running small experiments we support training and evaluation using a single GPU device.

### Preparing data

Our implementations of the algorithms require the data to be squared for 2D or cubic for 3D images.

### Clone the repository and install dependencies

Make sure you have [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Then perform the following commands, while you are in your desired workspace directory:
```shell script
git clone https://gitlab.com/statistical-genomics/self-supervised-3d-tasks.git
cd self-supervised-3d-tasks
conda env create -f environment.yml
conda activate conda-env
pip install -e .
```

### Running the experiments
To train a model with a specific algorithm run `python keras_train_algo.py configs/{algorithm}_{dimension}.json`

### Setting the configs
Training
Based on rotation_2d
```json 
{
  "algorithm": "'cpc'|'rotation'|'rpl'|'jigsaw'|'exemplar'",
  "batch_size": "Batch size as an integer.",
  "lr": "Learning rate as a float.",
  "epochs": "Amount of epochs as integer.",

  "dataset_name": "Name of the dataset, only used for labeling the log data.",
  "train3D": "Is the dataset 3D? As a boolean.",
  "data_dir": "The string representation of the path to of the data directory.",
  "data_dim": "Dimension of image as an integer.",
  "number_channels": "The number of channels of the image as integer.",

  "cpc, rpl": "",
  "patch_jitter": "CPC, RPL specific. Amount of pixels the jitter every patch should have as an integer.",
  "patches_per_side": "CPC, RPL specific. Amount of patches per dimension. 2 patches per side result in 8 patches for a 2D and 16 patches for a 3D image. As integer.",
  "crop_size": "CPC specific. ???",

  "train_data_generator_args": {
    "augment_zoom_only": true,
    "augment": true,
    "shuffle": true
  },
  "val_data_generator_args": {"augment": false},
  "encoder_architecture": "Name of the encoder architecture. Possibilities are 'DenseNet121'",
  "top_architecture": "Name of the top level architecture. Possibilities are 'big_fully'",
  "embed_dim": 1024,
  "save_checkpoint_every_n_epochs": "Backup epoch even without improvements every n epochs. As Integer.",
  "alpha_triplet": 0.2,
  "val_split": 0.05,
  "pooling": "none",
  "enc_filters": 8
}
```

Finetuning
Based on roation2d_finetunging
```json
{
  "algorithm": "'cpc'|'rotation'|'rpl'|'jigsaw'|'exemplar'",
  "lr": "Learning rate as a float.",
  "batch_size": "Batch size as an integer.",
  "val_split": 0.05,
  "epochs": "Amount of epochs as integer.",
  "repetitions": 3,
  "exp_splits": [100,10,1,50,25],


  "top_architecture": "big_fully",
  "encoder_architecture": "DenseNet121",
  "prediction_architecture": "simple_multiclass",
  "pooling": "avg",


  "dataset_name": "Name of the dataset, only used for labeling the log data.",
  "train3D": "Is the dataset 3D? As a boolean.",
  "data_dim": "Dimension of image as an integer.",
  "number_channels": "The number of channels of the image as integer.",
  "data_dir": "The string representation of the path to of the data directory.",
  "data_dir_train": "???",
  "data_dir_test": "???",
  "csv_file_train": "???",
  "csv_file_test": "???",
  "train_data_generator_args": {
    "suffix":  ".png",
    "multilabel": true,
    "augment": true,
    "shuffle": true
  },
  "val_data_generator_args": {
    "suffix":  ".png",
    "multilabel": true,
    "augment": false
  },
  "test_data_generator_args": {
    "suffix":  ".png",
    "multilabel": true,
    "augment": false
  },

  "metrics": ["accuracy"],
  "loss": "binary_crossentropy",
  "scores": ["qw_kappa_kaggle", "cat_acc_kaggle"],
  "clipnorm": 1,
  "clipvalue": 1,

  "embed_dim": 128,
  "epochs_warmup": 2,

  "load_weights": true,
  "model_checkpoint":"/home/Shared.Workspace/workspace/self-supervised-transfer-learning/shared_models/cpc_kaggle_retina/weights-250.hdf5",

  "patches_per_side": 4,
  "patch_dim": 64,
  "alpha_triplet": 0.2,
  "enc_filters": 8
}
```