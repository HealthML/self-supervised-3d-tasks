import functools
import os
import sys

import keras
from os.path import join, expanduser

from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d

from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from contextlib import redirect_stdout, redirect_stderr
import numpy as np

from self_supervised_3d_tasks.triplet_loss import triplet_loss_adapted_from_tf

aquire_free_gpus(1)
c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors

def preprocessing_exemplar(x, y, batch_id, nb_images, num_cases=4):
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
    # init np array for images
    x_processed = np.empty(shape=(batch_size, num_cases, x.shape[-3], x.shape[-2], x.shape[-1]))
    # init patch array
    patches = np.empty(shape=x.shape)
    # init np array with zeros
    y = np.zeros((nb_images, 1))
    # loop over all images with index and image
    for index, image in enumerate(x):
        # random transformation [0..1]
        random_lr_flip = np.random.randint(0, 2)
        random_ud_flip = np.random.randint(0, 2)
        distort_color = np.random.randint(0, 2)
        # flip up and down
        if random_ud_flip == 1:
            image = np.flip(image, 0)
        # flip left and right
        if random_lr_flip == 1:
            image = np.flip(image, 1)

        for case in range(num_cases):
            image = _distort_color(image)
            patches = np.append(patches, image)

        # TODO set label
        # set index
        y[batch_id*batch_size + index] = 1
        x_processed[index] = patches
    # return images and rotation
    return x, y


def train_model(epochs,
                dim,
                work_dir,
                data_dir,
                batch_size=8,
                lr=1e-3,
                n_channels=3,
                num_cases=1):
    """
    This method trains a resnet on Rotation task
    :param epochs: number of epochs
    :param dim: dimensions (without channels!)
    :param work_dir: path to save model checkpoints
    :param data_dir: path to images
    :param batch_size: batch size
    :param lr: learning rate
    :param n_channels: number of channels
    :param num_cases: defines the number of cases for pre processing
    :return:
    """

    num_classes = len(os.listdir(data_path))

    # init func
    func = functools.partial(preprocessing_exemplar, num_cases=num_cases, nb_images=num_classes)

    # init data generator
    train_data, validation_data = get_data_generators(data_dir, train_split=0.7,
                                                      train_data_generator_args={"batch_size": batch_size,
                                                                                 "dim": dim,
                                                                                 "n_channels": n_channels,
                                                                                 "pre_proc_func": func,
                                                                                 "exemplar": True},
                                                      test_data_generator_args={"batch_size": batch_size,
                                                                                "dim": dim,
                                                                                "n_channels": n_channels,
                                                                                "pre_proc_func": func,
                                                                                "exemplar": True}
                                                      )

    # compile model
    model = get_res_net_2d(input_shape=[*dim, n_channels], classes=num_classes, architecture="ResNet50", learning_rate=lr,
                           loss=triplet_loss_adapted_from_tf)

    # Callbacks
    tb_callback = keras.callbacks.TensorBoard(log_dir=work_dir,
                                              histogram_freq=0,
                                              batch_size=batch_size,
                                              write_graph=True, write_grads=False, write_images=False,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None,
                                              embeddings_data=None, update_freq='batch')

    mc_callback = keras.callbacks.ModelCheckpoint(
        work_dir + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
        save_best_only=False,
        save_weights_only=False, mode='auto', period=1)

    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-4),
                 tb_callback, mc_callback]

    # Train the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )


if __name__ == "__main__":
    # gpu_options = tf.GPUOptions()
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            # with tf.Session(config=config) as sess:
            working_dir = expanduser("~/workspace/self-supervised-transfer-learning/rotation_retina_192")
            data_path = "/mnt/mpws2019cl1/retinal_fundus/left/max_256/"
            number_channels = 3
            train_model(
                epochs=100,
                dim=(192, 192),
                batch_size=8,
                lr=1e-3,
                work_dir=working_dir,
                data_dir=data_path,
                n_channels=number_channels,
                num_cases=1
            )
