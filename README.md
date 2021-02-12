# 3D Self-Supervised Methods for Medical Imaging

Keras implementation of multiple self-supervised methods for 3D and 2D applications. This repository implements all the methods in this paper: [3D Self-Supervised Methods for Medical Imaging](https://arxiv.org/abs/2006.03829)

If you find this repository useful, please consider citing our paper in your work: 
```
@inproceedings{NEURIPS2020_d2dc6368,
 author = {Taleb, Aiham and Loetzsch, Winfried and Danz, Noel  and Severin, Julius and Gaertner, Thomas and Bergner, Benjamin and Lippert, Christoph},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {18158--18172},
 publisher = {Curran Associates, Inc.},
 title = {3D Self-Supervised Methods for Medical Imaging},
 url = {https://proceedings.neurips.cc/paper/2020/file/d2dc6368837861b42020ee72b0896182-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

## Overview

This codebase contains a implementation of five self-supervised representation learning
techniques, utility code for running training and evaluation loops.

## Usage instructions

In this codebase we provide configurations for training/evaluation of our models.

For debugging or running small experiments we support training and evaluation using a single GPU device.

### Preparing data

Our implementations of the algorithms require the data to be squared for 2D or cubic for 3D images.

### Clone the repository and install dependencies

Make sure you have [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Then perform the following commands, while you are in your desired workspace directory:
```shell script
git clone https://github.com/HealthML/self-supervised-3d-tasks.git
cd self-supervised-3d-tasks
conda env create -f env_all_platforms.yml
conda activate conda-env
pip install -e .
```

### Running the experiments
To train any of the self-supervised tasks with a specific algorithm, run `python train.py self_supervised_3d_tasks/configs/train/{algorithm}_{dimension}.json`
To run the downstream task and initialize the weights from a pretrained checkpoint, run `python finetune.py self_supervised_3d_tasks/configs/finetune/{algorithm}_{dimension}.json`

### Setting the configs

In the two example configs below, the respective parameters for training and testing configs are explained.

Training:
```json 
{
  "algorithm": "String. ('cpc'|'rotation'|'rpl'|'jigsaw'|'exemplar')",
  "batch_size": "Integer. Batch size.",
  "lr": "Float. Learning rate.",
  "epochs": "Integer. Amount of epochs as integer.",

  "encoder_architecture": "String. Name of the encoder architecture. ('DenseNet121'|'InceptionV3'|'ResNet50'|'ResNet50V2'|'ResNet101'|'ResNet101V2'|'ResNet152'|'InceptionResNetV2')",
  "top_architecture": "String. Name of the top level architecture. ('big_fully'|'simple_multiclass'|'unet_3d_upconv'|'unet_3d_upconv_patches') ",
    
  "dataset_name": "String. Name of the dataset, only used for labeling the log data.",
  "data_is_3D": "Boolean. Is the dataset 3D?.",
  "data_dir": "String. Path to of the data directory.",
  "data_dim": "Integer. Dimension of image.",
  "number_channels": "Integer. The number of channels of the image.",

  "patch_jitter": "Integer. CPC, RPL, Jigsaw specific. Amount of pixels the jitter every patch should have.",
  "patches_per_side": "Integer. CPC, RPL specific. Amount of patches per dimension. 2 patches per side result in 8 patches for a 2D and 16 patches for a 3D image.",
  "crop_size": "Integer. CPC specific. For CPC the whole image can be randomly cropped to a smaller size to make the self-supervised task harder",
  "code_size": "Integer. CPC, Exemplar specific. Specify the dimension of the latent space",
  
  "train_data_generator_args": {
    "suffix":  "String. ('.png'|'.jpeg')",
    "multilabel": "Boolean. Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Boolean. Include additional augmentations during loading the data. 2D augmentations: zooming, rotating. 3D augmentations: flipping, color distortion, rotation",
    "augment_zoom_only": "Boolean. 2D specific augmentations without rotating the image.",
    "shuffle": "Boolean. Shuffle the data after each epoch."
  },
  "val_data_generator_args": {
    "suffix":  "String. ('.png'|'.jpeg')",
    "multilabel": "Boolean. Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Boolean. Include additional augmentations during loading the data. 2D augmentations: zooming, rotating. 3D augmentations: flipping, color distortion, rotation",
    "augment_zoom_only": "Boolean. 2D specific augmentations without rotating the image.",
    "shuffle": "Boolean. Shuffle the data after each epoch."
  },

  "save_checkpoint_every_n_epochs": "Integer. Backup epoch even without improvements every n epochs.",
  "val_split": "Float between 0 and 1. Percentage of images used for test, None for no validation set.",
  "pooling": "String. (None|'avg'|'max')",
  "enc_filters": "Integer. Amount of filters used for the encoder model"
}
```

Finetuning:
```json
{
  "algorithm": "String. ('cpc'|'rotation'|'rpl'|'jigsaw'|'exemplar')",
  "lr": "Float. Learning rate.",
  "batch_size": "Integer. Batch size.",
  "val_split": "Float between 0 and 1. Percentage of images used for test. None for no validation set.",
  "epochs_warmup": "Integer. Amount of epochs used for warmup with frozen weights. ",
  "epochs": "Integer. Amount of epochs.",
  "repetitions": "Integer. Repetitions of the experiment.",
  "exp_splits": "Array<Integer>. Percentages of training data that should be used for the experiments. ([100,10,1,50,25])",


  "encoder_architecture": "String. Name of the encoder architecture. ('DenseNet121'|'InceptionV3'|'ResNet50'|'ResNet50V2'|'ResNet101'|'ResNet101V2'|'ResNet152'|'InceptionResNetV2')",
  "top_architecture": "String. Name of the top level architecture. ('big_fully'|'simple_multiclass'|'unet_3d_upconv'|'unet_3d_upconv_patches')",
  "prediction_architecture": "String. ('big_fully'|'simple_multiclass'|'unet_3d_upconv')",
  "pooling": "String. (None|'avg'|'max')",


  "dataset_name": "String. Name of the dataset, only used for labeling the log data.",
  "data_is_3D": "Boolean. Is the dataset 3D?.",
  "data_dim": "Integer. Dimension of image.",
  "number_channels": "Integer. The number of channels of the image.",
  "data_dir": "String. Path to the data directory the model was trained on.",
  "data_dir_train": "String. Path to the data directory containing the finetuning train data.",
  "data_dir_test": "String. Path to the data directory containing the finetuning test data.",
  "csv_file_train": "String. Path to the csv file containing the finetuning train data.",
  "csv_file_test": "String. Path to the csv file containing the finetuning test data.",
  "train_data_generator_args": {
    "suffix":  "String. ('.png'|'.jpeg')",
    "multilabel": "Boolean. Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Boolean. nclude additional augmentations during loading the data. 2D augmentations: zooming, rotating. 3D augmentations: flipping, color distortion, rotation.",
    "augment_zoom_only": "Boolean. 2D specific augmentations without rotating the image.",
    "shuffle": "Boolean. Shuffle the data after each epoch."
  },
  "val_data_generator_args": {
    "suffix":  "String. ('.png'|'.jpeg')",
    "multilabel": "Boolean. Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Boolean. Include additional augmentations during loading the data. 2D augmentations: zooming, rotating. 3D augmentations: flipping, color distortion, rotation",
    "augment_zoom_only": "Boolean. 2D specific augmentations without rotating the image.",
    "shuffle": "Boolean. Shuffle the data after each epoch."
  },
  "test_data_generator_args": {
    "suffix":  "String. ('.png'|'.jpeg')",
    "multilabel": "Boolean. Shall data be transformed to multilabel representation. (0 => [0, 0], 1 => [1, 0], 2 => [1, 1]",
    "augment": "Boolean. Include additional augmentations during loading the data. 2D augmentations: zooming, rotating. 3D augmentations: flipping, color distortion, rotation",
    "augment_zoom_only": "Boolean. 2D specific augmentations without rotating the image.",
    "shuffle": "Boolean. Shuffle the data after each epoch."
  },

  "metrics": "Array<String>. Metrics to be used. ('accuracy'|'mse')",
  "loss": "String. Loss to be used. ('binary_crossentropy'|'weighted_dice_loss'|'weighted_sum_loss'|'weighted_categorical_crossentropy'|'jaccard_distance')",
  "scores": "Array<String>. Scores to be used. ('qw_kappa'|'qw_kappa_kaggle'|'cat_accuracy'|'cat_acc_kaggle'|'dice'|'jaccard')",
  "clipnorm": "Float. Gradients will be clipped when their L2 norm exceeds this value.",
  "clipvalue": "Float. Gradients will be clipped when their absolute value exceeds this value.",

  "embed_dim": "Integer. Size of the embedding vector of the model.",

  "load_weights": "Boolean. Shall weights be loaded from model checkpoint.",
  "model_checkpoint":"String. Path to model checkpoint.",

  "patches_per_side": "Integer. CPC, RPL specific. Amount of patches per dimension. 2 patches per side result in 8 patches for a 2D and 16 patches for a 3D image.",
  "enc_filters": "Integer. Amount of filters used for the encoder model"
}
```

A sample labels csv file for the kaggle retina dataset can be found in: self_supervised_3d_tasks/data/example_labels_retina.csv 
