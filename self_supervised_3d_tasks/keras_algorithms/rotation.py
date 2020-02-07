from tensorflow.keras import  Input
from tensorflow.keras.layers import Flatten, TimeDistributed
from os.path import expanduser

from self_supervised_3d_tasks.custom_preprocessing.rotation import rotate_batch, resize
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# optionally load this from a config file at some time
data_dim = 192
n_channels = 3
data_shape = (data_dim, data_dim)
lr = 1e-3
image_size = data_dim
img_shape = (image_size, image_size, n_channels)
model_checkpoint = expanduser('~/workspace/self-supervised-transfer-learning/rotation_ukb_retina/weights-improvement-040.hdf5')


def get_training_model():
    model = get_res_net_2d(input_shape=img_shape, classes=4, architecture="ResNet50", learning_rate=lr)

    return model


def get_training_preprocessing():
    def f_train(x, y):  # not using y here, as it gets generated
        return rotate_batch(x, y)

    def f_val(x, y):
        return rotate_batch(x, y)

    return f_train, f_val


def get_finetuning_preprocessing():
    def f_train(x, y):
        return rotate_batch(x)[0], y

    def f_val(x, y):
        return rotate_batch(x)[0], y

    return f_train, f_val


def get_finetuning_layers(load_weights, freeze_weights):
    model = get_res_net_2d(input_shape=img_shape, classes=4, architecture="ResNet50", learning_rate=lr)

    if load_weights:
        model.load_weights(model_checkpoint)

    if freeze_weights:
        # freeze the encoder weights
        model.trainable = False

    layer_in = Input(img_shape)
    layer_out = TimeDistributed(model)(layer_in)

    x = Flatten()(layer_out)

    return layer_in, x, model
