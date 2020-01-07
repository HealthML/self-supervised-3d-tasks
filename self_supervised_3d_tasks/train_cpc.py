import math
import os
import sys

import keras
import keras.backend as k
from os.path import join, expanduser
from os import makedirs
from pathlib import Path
import numpy as np
import tensorflow as tf

from self_supervised_3d_tasks.data_util.cpc_utils import SortedNumberGenerator
from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc

from self_supervised_3d_tasks.algorithms.patch_model_preprocess import get_crop_patches_fn
from self_supervised_3d_tasks.preprocess import get_crop, get_random_flip_ud, get_drop_all_channels_but_one_preprocess, \
    get_pad, get_cpc_preprocess_grid
from self_supervised_3d_tasks.datasets import get_data, DatasetUKB

from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from contextlib import redirect_stdout, redirect_stderr

aquire_free_gpus(1)
c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors


def chain(f, g):
    return lambda x: g(f(x))


class PatchMatcher(object):
    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, is_training, session, batch_size):
        # return
        self.is_training = is_training
        self.batch_size = batch_size

        crop_size = 128
        split_per_side = 7
        patch_jitter = int(- crop_size / (split_per_side + 1))
        patch_crop_size = int((crop_size - patch_jitter * (split_per_side - 1)) / split_per_side * 7 / 8)
        padding = int((-2 * patch_jitter - patch_crop_size) / 2)

        f = lambda batch: batch
        f = chain(f, get_crop(is_training=self.is_training, crop_size=(crop_size, crop_size)))
        # f = chain(f, get_random_flip_ud(is_training=True)) also for new version?
        f = chain(f, get_crop_patches_fn(is_training=self.is_training, split_per_side=split_per_side,
                                         patch_jitter=patch_jitter))
        f = chain(f, get_random_flip_ud(is_training=self.is_training))
        f = chain(f, get_crop(is_training=self.is_training, crop_size=(patch_crop_size, patch_crop_size)))
        f = chain(f, get_drop_all_channels_but_one_preprocess())
        f = chain(f, get_pad([[padding, padding], [padding, padding], [0, 0]], "REFLECT"))

        # TODO put in when properly implemented
        # f = chain(f, get_cpc_preprocess_grid())

        data = DatasetUKB(
            split_name="train",
            preprocess_fn=f,
            num_epochs=1,
            shuffle=False,
            dataset_dir="/mnt/mpws2019cl1/brain_mri/tf_records/",
            random_seed=True,
            drop_remainder=True,
        ).input_fn({'batch_size': self.batch_size})

        self.iterator = data.make_one_shot_iterator()
        self.sess = session

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return math.ceil(DatasetUKB.COUNTS["train" if self.is_training else "val"] / self.batch_size)

    def next(self):
        el = self.iterator.get_next()
        batch = self.sess.run(el)
        batch = get_cpc_preprocess_grid()(batch)

        X = [np.array(batch["patches_enc"]), np.array(batch["patches_pred"])]
        Y = np.array(batch["labels"])
        print("XY", X[0].shape, X[1].shape, Y.shape)
        return (X, Y)


def train_model(epochs, code_size, lr=1e-4, terms=4, predict_terms=4, image_size=28, batch_size=8,
                session=None):
    working_dir = expanduser("~/workspace/self-supervised-transfer-learning/cpc")

    # Callbacks
    k.set_session(tf.Session(config=tf.ConfigProto(allow_soft_placement=True)))

    # Prepare data
    train_data = PatchMatcher(is_training=True, session=session, batch_size=batch_size)
    validation_data = PatchMatcher(is_training=False, session=session, batch_size=batch_size)

    # Prepares the model
    model = network_cpc(image_shape=(image_size, image_size, 2), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=lr)

    tb_callback = keras.callbacks.TensorBoard(log_dir=working_dir, histogram_freq=0,
                                              batch_size=batch_size,
                                              write_graph=True, write_grads=False, write_images=False,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None,
                                              embeddings_data=None, update_freq='epoch')

    mc_callback = keras.callbacks.ModelCheckpoint(working_dir, monitor='val_loss', verbose=0,
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

    # # Saves the model
    # # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    # path = Path(__file__).parent / output_dir
    # if not path.exists():
    #     makedirs(path)
    #
    # model.save(path / 'cpc.h5')
    #
    # # Saves the encoder alone
    # encoder = model.layers[1].layer
    # encoder.save(path / 'encoder.h5')


if __name__ == "__main__":
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            with tf.Session(config=config) as sess:
                train_model(
                    epochs=10,
                    code_size=128,
                    lr=1e-3,
                    terms=3,
                    predict_terms=3,
                    image_size=32,
                    session=sess,
                    batch_size=8
                )
