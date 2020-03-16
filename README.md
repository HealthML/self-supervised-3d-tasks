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
To train any of the self-supervised tasks with a specific algorithm, run `python train.py configs/train/{algorithm}_{dimension}.json`
To run the downstream task and initialize the weights from a pretrained checkpoint, run `python test.py configs/test/{algorithm}_{dimension}.json`

### Setting the configs

In the two example configs below, the respective parameters for training and testing configs are explained.

Training:
```json 
{
  "algorithm": "'cpc'|'rotation'|'rpl'|'jigsaw'|'exemplar'",
  "batch_size": "Batch size as an integer.",
  "lr": "Learning rate as a float.",
  "epochs": "Amount of epochs as integer.",

  "encoder_architecture": "Name of the encoder architecture. ('DenseNet121'|'InceptionV3'|'ResNet50'|'ResNet50V2'|'ResNet101'|'ResNet101V2'|'ResNet152'|'InceptionResNetV2')",
  "top_architecture": "Name of the top level architecture. ('big_fully'|'simple_multiclass'|'unet_3d_upconv'|'unet_3d_upconv_patches') ",
    
  "dataset_name": "Name of the dataset, only used for labeling the log data.",
  "train3D": "Is the dataset 3D? As a boolean.",
  "data_dir": "The string representation of the path to of the data directory.",
  "data_dim": "Dimension of image as an integer.",
  "number_channels": "The number of channels of the image as integer.",

  "cpc, rpl": "",
  "patch_jitter": "CPC, RPL, jigsaw specific. Amount of pixels the jitter every patch should have as an integer.",
  "patches_per_side": "CPC, RPL specific. Amount of patches per dimension. 2 patches per side result in 8 patches for a 2D and 16 patches for a 3D image. As integer.",
  "crop_size": "CPC specific. For CPC the whole image can be randomly cropped to a smaller size to make the self-supervised task harder",

  "train_data_generator_args": {
    "augment_zoom_only": "2D specific augmentations without rotating the image",
    "augment": "Include additional augmentations during loading the data. 2D augmentations: zooming, rotating. 3D augmentations: flipping, color distortion, rotation",
    "shuffle": "Shuffle the data after each epoch"
  },
  "val_data_generator_args": {"augment": false},
  "code_size": "CPC, Exemplar specific. Specify the dimension of the latent space",
  "save_checkpoint_every_n_epochs": "Backup epoch even without improvements every n epochs. As Integer.",
  "alpha_triplet": 0.2,
  "val_split": 0.05,
  "pooling": "none",
  "enc_filters": 8
}
```

Testing:
```json
{
  "algorithm": "'cpc'|'rotation'|'rpl'|'jigsaw'|'exemplar'",
  "lr": "Learning rate as a float.",
  "batch_size": "Batch size as an integer.",
  "val_split": "Percentage of images used for test, None for no validation set. Float between 0 and 1.",
  "epochs_warmup": 2,
  "epochs": "Amount of epochs as integer.",
  "repetitions": "Repetitions of the experiment as integer.",
  "exp_splits": "Percentages of training data that should be used for the experiments. Array of integers ([100,10,1,50,25])",


  "encoder_architecture": "Name of the encoder architecture. ('DenseNet121'|'InceptionV3'|'ResNet50'|'ResNet50V2'|'ResNet101'|'ResNet101V2'|'ResNet152'|'InceptionResNetV2')",
  "top_architecture": "Name of the top level architecture. ('big_fully'|'simple_multiclass'|'unet_3d_upconv'|'unet_3d_upconv_patches') ",
  "prediction_architecture": "simple_multiclass",
  "pooling": "avg|max",


  "dataset_name": "Name of the dataset, only used for labeling the log data.",
  "train3D": "Is the dataset 3D? As a boolean.",
  "data_dim": "Dimension of image as an integer.",
  "number_channels": "The number of channels of the image as integer.",
  "data_dir": "Path to the data directory the model was trained on as a string",
  "data_dir_train": "Path to the data directory containing the finetuning train data as a string.",
  "data_dir_test": "Path to the data directory containing the finetuning test data as a string.",
  "csv_file_train": "Path to the csv file containing the finetuning train data as a string.",
  "csv_file_test": "Path to the csv file containing the finetuning test data as a string.",
  "train_data_generator_args": {
    "suffix":  "(.png|.jpeg)",
    "multilabel": "Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Shall the data be randomly augmented with horizontally/vertically flip and/or zoom. Boolean.",
    "augment_zoom_only": "Shall the data be augmented with zoom. Boolean.",
    "shuffle": "Shall the data be shuffled after each epoch. Boolean."
  },
  "val_data_generator_args": {
    "suffix":  "(.png|.jpeg)",
    "multilabel": "Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Shall the data be randomly augmented with horizontally/vertically flip and/or zoom.",
    "augment_zoom_only": "Shall the data be augmented with zoom.",
  },
  "test_data_generator_args": {
    "suffix":  "(.png|.jpeg)",
    "multilabel": "Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Shall the data be randomly augmented with horizontally/vertically flip and/or zoom.",
    "augment_zoom_only": "Shall the data be augmented with zoom.",
  },

  "metrics": "Array of metrics to be used. Array of strings. ('accuracy'|'mse')",
  "loss": "Loss to be used. String('binary_crossentropy'|'weighted_dice_loss'|'weighted_sum_loss'|'weighted_categorical_crossentropy'|'jaccard_distance')",
  "scores": "Array of scores to be used. Array of strings. ('qw_kappa'|'qw_kappa_kaggle'|'cat_accuracy'|'cat_acc_kaggle'|'dice'|'jaccard')",
  "clipnorm": "Gradients will be clipped when their L2 norm exceeds this value.",
  "clipvalue": "Gradients will be clipped when their absolute value exceeds this value.",

  "embed_dim": "Size of the embedding vector of the model. Integer",

  "load_weights": "Shall weights be loaded from model checkpoint. Boolean.",
  "model_checkpoint":"Path to model checkpoint. String.",

  "patches_per_side": "CPC, RPL specific. Amount of patches per dimension. 2 patches per side result in 8 patches for a 2D and 16 patches for a 3D image. As integer.",
  "alpha_triplet": "The alpha value used for triplet loss.",
  "enc_filters": 8
}
```