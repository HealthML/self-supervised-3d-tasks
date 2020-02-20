import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm


def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    # im = im.resize((desired_size,) * 2, resample=Image.LANCZOS)

    return im


def get_dataset_kaggle_train(batch_size, *params, **kwargs):
    train_df = pd.read_csv('/mnt/mpws2019cl1/kaggle_retina_2019/train.csv')
    test_df = pd.read_csv('/mnt/mpws2019cl1/kaggle_retina_2019/test.csv')
    print(train_df.shape)
    print(test_df.shape)
    train_df.head()

    N = train_df.shape[0]
    x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

    for i, image_id in enumerate(tqdm(train_df['id_code'])):
        x_train[i, :, :, :] = preprocess_image(
            f'/mnt/mpws2019cl1/kaggle_retina_2019/images/resized_224/{image_id}.png'
        )

    # N = test_df.shape[0]
    # x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)
    #
    # for i, image_id in enumerate(tqdm(test_df['id_code'])):
    #     x_test[i, :, :, :] = preprocess_image(
    #         f'/mnt/mpws2019cl1/kaggle_retina_2019/images/resized_224/{image_id}.png'
    #     )

    y_train = pd.get_dummies(train_df['diagnosis']).values

    print(x_train.shape)
    print(y_train.shape)
    #print(x_test.shape)

    y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
    y_train_multi[:, 4] = y_train[:, 4]

    for i in range(3, -1, -1):
        y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i + 1])

    print(y_train[:, 4])
    print(y_train_multi)

    print("Original y_train:", y_train.sum(axis=0))
    print("Multilabel version:", y_train_multi.sum(axis=0))

    # x_train, x_val, y_train, y_val = train_test_split(
    #     x_train, y_train_multi,
    #     test_size=0.15,
    #     random_state=2019
    # )

    y_train = y_train_multi

    def create_datagen():
        return ImageDataGenerator(
            zoom_range=0.15,  # set range for random zoom
            # set mode for filling points outside the input boundaries
            fill_mode='constant',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
        )

    # Using original generator
    data_generator = create_datagen().flow(x_train, y_train, batch_size=batch_size, shuffle=False)
    # data_generator_val = ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size)

    return data_generator  #, data_generator_val
