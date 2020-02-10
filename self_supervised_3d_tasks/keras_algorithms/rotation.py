from keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, TimeDistributed
from os.path import expanduser

from self_supervised_3d_tasks.custom_preprocessing.rotation import rotate_batch, resize
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# model_checkpoint = expanduser(
#     "~/workspace/self-supervised-transfer-learning/rotation_ukb_retina/weights-improvement-040.hdf5"
# )


class RotationBuilder:
    def __init__(
            self, data_dim=192, n_channels=3, lr=1e-3, **kwargs
    ):
        self.data_dim = data_dim
        self.n_channels = n_channels
        self.lr = lr
        self.data_shape = (data_dim, data_dim)
        self.image_size = data_dim
        self.img_shape = (self.image_size, self.image_size, n_channels)

    def get_training_model(self):
        model = get_res_net_2d(
            input_shape=self.img_shape,
            classes=4,
            architecture="ResNet50",
            learning_rate=lr,
        )

        return model

    def get_training_preprocessing(self):
        def f_train(x, y):  # not using y here, as it gets generated
            return rotate_batch(x, y)

        def f_val(x, y):
            return rotate_batch(x, y)

        return f_train, f_val

    def get_finetuning_preprocessing(self):
        def f_train(x, y):
            return rotate_batch(x)[0], y

        def f_val(x, y):
            return rotate_batch(x)[0], y

        return f_train, f_val

    def get_finetuning_layers(self, load_weights, freeze_weights, model_checkpoint):
        model = get_res_net_2d(
            input_shape=self.img_shape,
            classes=4,
            architecture="ResNet50",
            learning_rate=self.lr,
        )

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        layer_in = Input(self.img_shape)
        layer_out = TimeDistributed(model)(layer_in)

        x = Flatten()(layer_out)

        return Model(layer_in, x), [model]


def create_instance(*params, **kwargs):
    return RotationBuilder(*params, **kwargs)
