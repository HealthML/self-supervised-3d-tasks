
import sys

import tensorflow.keras as keras
from os.path import join, expanduser

from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d

from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from contextlib import redirect_stdout, redirect_stderr
import numpy as np

aquire_free_gpus(1)
c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors


def rotation_2d(x,y=None, nb_images=None):
    """
    This function preprocess a batch for relative patch location in a 2 dimensional space.
    :param x: array of images
    :param y: None
    :return: x as np.array of images with random rotations, y np.array with one-hot encoded label
    """
    # get batch size
    batch_size = len(y)
    # init np array with zeros
    y = np.zeros((batch_size, 4))
    # loop over all images with index and image
    for index, image in enumerate(x):
        # random transformation [0..3]
        rot = np.random.random_integers(4) - 1
        # iterate over rotations
        for i in range(0, rot):
            # rotate the image
            image = np.rot90(image)
        # set image
        x[index] = image
        # set index
        y[index, rot] = 1
    # return images and rotation
    return x,y


def train_model(epochs,
                dim,
                work_dir,
                data_dir,
                batch_size=8,
                lr=1e-3,
                n_channels=3):
    """
    This method trains a resnet on Rotation task
    :param epochs: number of epochs
    :param dim: dimensions (without channels!)
    :param work_dir: path to save model checkpoints
    :param data_dir: path to images
    :param batch_size: batch size
    :param lr: learning rate
    :param n_channels: number of channels
    :return:
    """
    # init data generator
    train_data, validation_data = get_data_generators(data_dir, train_split=0.7,
                                                      train_data_generator_args={"batch_size": batch_size,
                                                                                 "dim": dim,
                                                                                 "n_channels": n_channels,
                                                                                 "pre_proc_func": rotation_2d},
                                                      test_data_generator_args={"batch_size": batch_size,
                                                                                "dim": dim,
                                                                                "n_channels": n_channels,
                                                                                "pre_proc_func": rotation_2d}
                                                      )


    # compile model
    model = get_res_net_2d(input_shape=[*dim, n_channels], classes=4, architecture="ResNet50", learning_rate=lr)

    # Callbacks
    tb_callback = keras.callbacks.TensorBoard(log_dir=work_dir, histogram_freq=0,
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
            working_dir = expanduser("~/workspace/self-supervised-transfer-learning/rotation_retina_512")
            data_path = "/mnt/mpws2019cl1/retinal_fundus/left/max_512"
            number_channels = 3
            train_model(
                epochs=100,
                dim=(512, 512),
                batch_size=4,
                lr=1e-4,
                work_dir=working_dir,
                data_dir=data_path,
                n_channels=number_channels
            )

