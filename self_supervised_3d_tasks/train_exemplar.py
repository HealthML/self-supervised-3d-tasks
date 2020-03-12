from datetime import datetime
import functools
import os
import sys
import time

import keras
from os.path import join, expanduser

from self_supervised_3d_tasks.data.data_generator import get_data_generators

from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from self_supervised_3d_tasks.models.exemplar_model import get_exemplar_model

aquire_free_gpus(1)
c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors


def preprocessing_exemplar(x, y, process_3d = False):
    def _distort_color(scan):
        """
        This function is based on the distort_color function from the tf implementation.
        :param scan: image as np.array
        :return: processed image as np.array
        """
        # TODO distort colors
        # adjust brightness
        max_delta = 32.0 / 255.0
        delta = np.random.uniform(-max_delta, max_delta)
        scan += delta

        # adjust contrast
        lower = 0.5
        upper = 1.5
        contrast_factor = np.random.uniform(lower, upper)
        scan_mean = np.mean(scan)
        scan = (contrast_factor * (scan - scan_mean)) + scan_mean
        return scan

    """
    This function preprocess a batch for relative patch location in a 2 dimensional space.
    :param x: array of images
    :param y: None
    :return: x as np.array of images with random rotations, y np.array with one-hot encoded label
    """
    # get batch size
    batch_size = len(y)
    if process_3d:
        x_processed = np.empty(shape=(batch_size, 3, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]))
    else:
        x_processed = np.empty(shape=(batch_size, 3, x.shape[-3], x.shape[-2], x.shape[-1]))
    # init patch array
    if process_3d:
        triplet = np.empty(shape=(3, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]))
    else:
        triplet = np.empty(shape=(3, x.shape[-3], x.shape[-2], x.shape[-1]))
    # get negative examples
    random_images = x
    np.random.shuffle(random_images)
    # loop over all images with index and image
    for index, image in enumerate(x):
        # random transformation [0..1]
        random_lr_flip = np.random.randint(0, 2)
        random_ud_flip = np.random.randint(0, 2)
        distort_color = np.random.randint(0, 2)
        processed_image = image
        # flip up and down
        if random_ud_flip == 1:
            processed_image = np.flip(processed_image, 0)
        # flip left and right
        if random_lr_flip == 1:
            processed_image = np.flip(processed_image, 1)
        # distort_color
        if distort_color == 1:
            processed_image = _distort_color(processed_image)

        # Set Anchor Image
        triplet[0] = processed_image
        # Set Positiv Image
        triplet[1] = image
        # Set negativ Image
        negativ_image = random_images[index]
        triplet[2] = negativ_image
        x_processed[index] = triplet
    # return images and rotation
    return x_processed, y


def train_model(epochs,
                dim,
                work_dir,
                data_dir,
                batch_size=8,
                number_channels=3,
                dim_3d=True,
                model_params={}):
    """
    This method trains a Encoder Model on the exemplar task
    :param epochs: number of epochs
    :param dim: dimensions (without channels!)
    :param work_dir: path to save model checkpoints
    :param data_dir: path to images
    :param batch_size: batch size
    :param model_params: additional Params for the model
    :param number_channels: number of channels
    :param dim_3d: defines whether 2d or 3d is used
    :return:
    """

    # init func
    func = functools.partial(preprocessing_exemplar, process_3d=dim_3d)

    # init data generator
    train_data, test_data = get_data_generators(data_dir, train_split=0.7,
                                                train_data_generator_args={"batch_size": batch_size,
                                                                           "dim": dim,
                                                                           "pre_proc_func": func},
                                                test_data_generator_args={"batch_size": batch_size,
                                                                          "dim": dim,
                                                                          "pre_proc_func": func}
                                                )

    # compile model
    # TODO Generate Model
    model = get_exemplar_model(input_shape=(*dim, number_channels), **model_params)
    print(model.summary())

    # Train the model
    model = train_loop(model, train_data, test_data, work_dir, epochs, validation_every=500)
    model.save_weights("{}/final_model.h5".format(work_dir))
    return 0


def keras_callback(val_loss, loss, step_number, working_dir):
    items_to_write = {
        "loss": loss,
        "val_loss": val_loss
    }
    writer = tf.summary.FileWriter(working_dir)
    for name, value in items_to_write.items():
        summary = tf.summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, step_number)
        writer.flush()


def train_loop(model, train_data, test_data, work_dir, epochs=100, validation_every=500):
    """
    :param model: model for training
    :param train_data: train data generator
    :param test_data: test data generator
    :param work_dir: working directory
    :param epochs: number of epochs
    :param validation_every: validation
    :return: model
    """
    step_number = 0
    train_loss = 0
    val_loss = 0
    loss = []
    for epoch in range(0, epochs):
        print("Epoch {}".format(epoch))
        for iteration in tqdm(range(0, train_data.__len__())):
            triplets, _ = train_data.__getitem__(iteration)
            loss.append(model.train_on_batch(triplets, None))
            step_number += 1
            if step_number % validation_every == 0:
                train_loss = np.mean(np.asarray(loss))
                val_loss = validate_model(model, test_data)
                keras_callback(val_loss, train_loss, step_number, work_dir)
                loss = []
        print("Loss: {}".format(train_loss))
        print("Val_Loss: {}".format(val_loss))
        model.save_weights("{}/exemplar-{}-{}.h5".format(work_dir, epoch, val_loss))
    return model


def validate_model(model, test_data):
    """
    Validate model on test_data
    :param model: triplet loss model
    :param test_data: test data generator
    :return: test loss
    """
    n_iteration = 0
    loss = []
    for iteration in tqdm(range(0, test_data.__len__())):
        triplets, _ = test_data.__getitem__(iteration)
        loss.append(model.test_on_batch(triplets, None))
        n_iteration += 1
    return np.mean(np.asarray(loss))


if __name__ == "__main__":
    # gpu_options = tf.GPUOptions()
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            # with tf.Session(config=config) as sess:
            working_dir = expanduser("~/workspace/self-supervised-transfer-learning/exemplar_ResNet")
            data_path = "/mnt/mpws2019cl1/retinal_fundus/left/max_256"
            number_channels = 3
            train_model(
                epochs=100,
                dim=(256, 256),
                batch_size=4,
                work_dir=working_dir,
                data_dir=data_path,
                number_channels=number_channels,
                dim_3d=False,
                model_params={
                    "alpha_triplet": 0.2,
                    "embedding_size": 10,
                    "lr": 0.0006
                }
            )
