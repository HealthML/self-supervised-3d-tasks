from os.path import expanduser

import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Dense, TimeDistributed
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import preprocess_grid, preprocess, resize
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms.cpc_model_utils import network_cpc
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator

# optionally load this from a config file at some time
data_dir = "/mnt/mpws2019cl1/kaggle_retina/train/resized_384"
data_dim = 384  # original data dim
data_shape = (data_dim, data_dim)
crop_size = 384
split_per_side = 7
n_channels = 3
code_size = 128
lr = 1e-3
terms = 9
predict_terms = 3
image_size = int((crop_size / (split_per_side + 1)) * 2)
img_shape = (image_size, image_size, n_channels)
model_checkpoint = expanduser('~/workspace/self-supervised-transfer-learning/cpc_kaggle_retina_12/weights-improvement-011.hdf5')


def get_training_model():
    model, enc_model = network_cpc(image_shape=img_shape, terms=terms,
                                   predict_terms=predict_terms,
                                   code_size=code_size, learning_rate=lr)
    # we get a model that is already compiled
    return model


def get_training_preprocessing():
    def f_train(x, y):  # not using y here, as it gets generated
        return preprocess_grid(preprocess(x, crop_size, split_per_side))

    def f_val(x, y):
        return preprocess_grid(preprocess(x, crop_size, split_per_side, is_training=False))

    return f_train, f_val


def get_finetuning_preprocessing():
    def f_train(x, y):
        return preprocess(resize(x, data_dim), crop_size, split_per_side, is_training=False), y

    def f_val(x, y):
        return preprocess(resize(x, data_dim), crop_size, split_per_side, is_training=False), y

    return f_train, f_val


def get_finetuning_layers(load_weights, freeze_weights):
    cpc_model, enc_model = network_cpc(image_shape=img_shape, terms=terms, predict_terms=predict_terms,
                                       code_size=code_size, learning_rate=lr)

    if load_weights:
        cpc_model.load_weights(model_checkpoint)

    if freeze_weights:
        # freeze the encoder weights
        enc_model.trainable = False

    layer_in = Input((split_per_side * split_per_side,) + img_shape)
    layer_out = TimeDistributed(enc_model)(layer_in)

    x = Flatten()(layer_out)
    return layer_in, x, [enc_model, cpc_model]
