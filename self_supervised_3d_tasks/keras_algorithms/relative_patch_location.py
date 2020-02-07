from keras import Model, Input
from keras.layers import Flatten, Dense, TimeDistributed
from os.path import expanduser

from self_supervised_3d_tasks.custom_preprocessing.relative_patch_location import preprocess_batch, resize
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# optionally load this from a config file at some time
data_dim = 192
n_channels = 3
data_shape = (data_dim, data_dim)
patches_per_side = 3
patch_jitter = 24
lr = 1e-3
image_size = int(data_dim / patches_per_side) - patch_jitter
img_shape = (image_size, image_size, n_channels)
model_checkpoint = expanduser('~/workspace/self-supervised-transfer-learning/rpl_ukb_retina/'
    'weights-improvement-98-0.98.hdf5')



def get_training_model():
    model = get_res_net_2d(input_shape=(image_size, image_size, n_channels), classes=patches_per_side**2, architecture="ResNet50", learning_rate=lr)

    return model


def get_training_preprocessing():
    def f_train(x, y):  # not using y here, as it gets generated
        return preprocess_batch(x, patches_per_side, patch_jitter)

    def f_val(x, y):
        return preprocess_batch(x, patches_per_side, patch_jitter)

    return f_train, f_val


def get_finetuning_preprocessing():
    def f_train(x, y):
        return preprocess_batch(resize(x, data_dim), patches_per_side, 0, is_training=False)[0], y

    def f_val(x, y):
        return preprocess_batch(resize(x, data_dim), patches_per_side, 0, is_training=False)[0], y

    return f_train, f_val


def get_finetuning_layers(load_weights, freeze_weights):
    model = get_res_net_2d(input_shape=[image_size, image_size, n_channels], classes=patches_per_side**2, architecture="ResNet50", learning_rate=lr)

    if load_weights:
        model.load_weights(model_checkpoint)

    if freeze_weights:
        # freeze the encoder weights
        model.trainable = False

    layer_in = Input((patches_per_side*patches_per_side,) + img_shape)
    layer_out = TimeDistributed(model)(layer_in)

    x = Flatten()(layer_out)

    return layer_in, x, model