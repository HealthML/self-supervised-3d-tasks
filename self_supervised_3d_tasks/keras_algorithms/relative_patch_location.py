from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Dense, TimeDistributed
from os.path import expanduser

from self_supervised_3d_tasks.custom_preprocessing.relative_patch_location import (
    preprocess_batch,
    resize,
)
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# optionally load this from a config file at some time

# model_checkpoint = expanduser(
#     "~/workspace/self-supervised-transfer-learning/rpl_ukb_retina/"
#     "weights-improvement-98-0.98.hdf5"
# )


class RelativePatchLocationBuilder:
    def __init__(
            self, data_dim=192, n_channels=3, patches_per_side=3, patch_jitter=24, lr=1e-3,
    ):
        self.data_dim = data_dim
        self.n_channels = n_channels
        self.patches_per_side = patches_per_side
        self.patch_jitter = patch_jitter
        self.lr = lr
        self.data_shape = (data_dim, data_dim)
        self.image_size = int(data_dim / patches_per_side) - patch_jitter
        self.img_shape = (self.image_size, self.image_size, n_channels)
        pass

    def get_training_model(self):
        model = get_res_net_2d(
            input_shape=(self.image_size, self.image_size, self.n_channels),
            classes=self.patches_per_side ** 2,
            architecture="ResNet50",
            learning_rate=self.lr,
        )

        return model

    def get_training_preprocessing(self):
        def f_train(x, y):  # not using y here, as it gets generated
            return preprocess_batch(x, self.patches_per_side, self.patch_jitter)

        def f_val(x, y):
            return preprocess_batch(x, self.patches_per_side, self.patch_jitter)

        return f_train, f_val

    def get_finetuning_preprocessing(self):
        def f_train(x, y):
            return (
                preprocess_batch(
                    resize(x, self.data_dim),
                    self.patches_per_side,
                    0,
                    is_training=False,
                )[0],
                y,
            )

        def f_val(x, y):
            return (
                preprocess_batch(
                    resize(x, self.data_dim),
                    self.patches_per_side,
                    0,
                    is_training=False,
                )[0],
                y,
            )

        return f_train, f_val

    def get_finetuning_layers(self, load_weights, freeze_weights, model_checkpoint):
        model = get_res_net_2d(
            input_shape=[self.image_size, self.image_size, self.n_channels],
            classes=self.patches_per_side ** 2,
            architecture="ResNet50",
            learning_rate=self.lr,
        )

        if load_weights:
            model.load_weights(model_checkpoint)

        if freeze_weights:
            # freeze the encoder weights
            model.trainable = False

        layer_in = Input((self.patches_per_side * self.patches_per_side,) + self.img_shape)
        layer_out = TimeDistributed(model)(layer_in)

        x = Flatten()(layer_out)

        return layer_in, x, model


def create_instance(*params, **kwargs):
    RelativePatchLocationBuilder(*params, **kwargs)
