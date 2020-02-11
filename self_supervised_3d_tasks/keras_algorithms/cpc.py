from os.path import expanduser

import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Dense, TimeDistributed
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import (
    preprocess_grid,
    preprocess,
    resize,
)
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms.cpc_model_utils import network_cpc
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# optionally load this from a config file at some time
# data_dir = "/mnt/mpws2019cl1/kaggle_retina/train/resized_384"
# model_checkpoint = expanduser(
#     "~/workspace/self-supervised-transfer-learning/cpc_kaggle_retina_12/weights-improvement-011.hdf5"
# )


class CPCBuilder:
    def __init__(
            self,
            data_shape=(384, 384, 3),
            data_dim=384,
            crop_size=int(384*0.95),
            split_per_side=7,
            code_size=128,
            lr=1e-3,
            terms=9,
            predict_terms=3,
            channels_last=True,
            **kwargs,
    ):
        self.data_shape = data_shape
        self.data_dim = data_dim
        self.n_channels = data_shape[-1] if channels_last else data_shape[0]
        self.crop_size = crop_size
        self.split_per_side = split_per_side
        self.code_size = code_size
        self.lr = lr
        self.terms = terms
        self.predict_terms = predict_terms
        self.image_size = int((data_dim / (split_per_side + 1)) * 2)
        self.img_shape = (self.image_size, self.image_size, self.n_channels)
        self.kwargs = kwargs

    def get_training_model(self):
        model, enc_model = network_cpc(
            image_shape=self.img_shape,
            terms=self.terms,
            predict_terms=self.predict_terms,
            code_size=self.code_size,
            learning_rate=self.lr,
        )
        # we get a model that is already compiled
        return model

    def get_training_preprocessing(self):
        def f_train(x, y):  # not using y here, as it gets generated
            return preprocess_grid(preprocess(x, self.crop_size, self.split_per_side))

        def f_val(x, y):
            return preprocess_grid(
                preprocess(x, self.crop_size, self.split_per_side, is_training=False)
            )

        return f_train, f_val

    def get_finetuning_preprocessing(self):
        def f_train(x, y):
            return (
                preprocess(
                    resize(x, self.data_dim),
                    self.crop_size,
                    self.split_per_side,
                    is_training=False,
                ),
                y,
            )

        def f_val(x, y):
            return (
                preprocess(
                    resize(x, self.data_dim),
                    self.crop_size,
                    self.split_per_side,
                    is_training=False,
                ),
                y,
            )

        return f_train, f_val

    def get_finetuning_model(self, model_checkpoint=None):
        cpc_model, enc_model = network_cpc(
            image_shape=self.img_shape,
            terms=self.terms,
            predict_terms=self.predict_terms,
            code_size=self.code_size,
            learning_rate=self.lr,
        )

        if model_checkpoint is not None:
            cpc_model.load_weights(model_checkpoint)

        layer_in = Input((self.split_per_side * self.split_per_side,) + self.img_shape)
        layer_out = TimeDistributed(enc_model)(layer_in)

        x = Flatten()(layer_out)
        return Model(layer_in, x), [enc_model, cpc_model]


def create_instance(*params, **kwargs):
    return CPCBuilder(*params, **kwargs)
