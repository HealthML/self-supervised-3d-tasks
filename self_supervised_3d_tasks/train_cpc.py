import sys
from contextlib import redirect_stdout, redirect_stderr
from os.path import expanduser

import os
import keras

from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import preprocess, preprocess_grid
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors


def train_model(epochs, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, batch_size=8):
    working_dir = expanduser("~/workspace/self-supervised-transfer-learning/cpc_retina")
    data_dir = "/mnt/mpws2019cl1/retinal_fundus/left/max_256/"

    crop_size = 186
    split_per_side = 7
    n_channels = 3

    f_train = lambda x, y: preprocess_grid(preprocess(x, crop_size, split_per_side))
    f_val = lambda x, y: preprocess_grid(preprocess(x, crop_size, split_per_side, is_training=False))

    train_data, validation_data = get_data_generators(data_dir, train_split=0.7,
                                                      train_data_generator_args={"batch_size": batch_size,
                                                                                 "dim": (192, 192),
                                                                                 "n_channels": n_channels,
                                                                                 "pre_proc_func": f_train},
                                                      test_data_generator_args={"batch_size": batch_size,
                                                                                "dim": (192, 192),
                                                                                "n_channels": n_channels,
                                                                                "pre_proc_func": f_val}
                                                      )

    # Prepare data
    # train_data = PatchMatcher(is_training=True, session=session, batch_size=batch_size)
    # validation_data = PatchMatcher(is_training=False, session=session, batch_size=batch_size)

    # Prepares the model
    model, _ = network_cpc(image_shape=(image_size, image_size, n_channels), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

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
                code_size=128,
                lr=1e-3,
                terms=3,
                predict_terms=3,
                image_size=46,
                batch_size=8
            )
