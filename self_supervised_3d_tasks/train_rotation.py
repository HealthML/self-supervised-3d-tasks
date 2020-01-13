
import sys

import keras
from os.path import join, expanduser

from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d

from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from contextlib import redirect_stdout, redirect_stderr
import numpy as np

aquire_free_gpus(1)
c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors


def rotation_2d(x,y):
    batch_size = len(y)
    y = np.zeros((batch_size, 4))
    for i, image in enumerate(x):
        rot = np.random.random_integers(4) -1
        for i in range(i, rot):
            image = np.rot90(image)
        x[i] = image
        y[i, rot] = 1
        return x,y


def train_model(epochs, dim, batch_size=8, lr=1e-3):
    working_dir = expanduser("~/workspace/self-supervised-transfer-learning/rotation_retina")
    data_dir = "/mnt/mpws2019cl1/retinal_fundus/left/max_256/"
    n_channels = 3

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

    # Prepare data
    # train_data = PatchMatcher(is_training=True, session=session, batch_size=batch_size)
    # validation_data = PatchMatcher(is_training=False, session=session, batch_size=batch_size)

    # Prepares the model
    model = get_res_net_2d(input_shape=[*dim, n_channels], classes=4, architecture="ResNet50", learning_rate=lr)

    tb_callback = keras.callbacks.TensorBoard(log_dir=working_dir, histogram_freq=0,
                                              batch_size=batch_size,
                                              write_graph=True, write_grads=False, write_images=False,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None,
                                              embeddings_data=None, update_freq='batch')

    mc_callback = keras.callbacks.ModelCheckpoint(
        working_dir + "/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='val_loss', verbose=0,
        save_best_only=False,
        save_weights_only=False, mode='auto', period=1)

    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-4),
                 tb_callback, mc_callback]

    # Trains the model
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
            train_model(
                epochs=10,
                dim=(192, 192),
                batch_size=8,
                lr=1e-3,
            )

