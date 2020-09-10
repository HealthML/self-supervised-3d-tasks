import numpy as np
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_rpl import (
    preprocess_batch,
    preprocess_batch_3d
)
from self_supervised_3d_tasks.utils.model_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model)


class RelativePatchLocationBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            patches_per_side=3,
            patch_jitter=0,
            lr=1e-3,
            data_is_3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        super(RelativePatchLocationBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.patch_jitter = patch_jitter
        self.top_architecture = top_architecture

        self.patches_per_side = patches_per_side
        self.patch_dim = (data_dim // patches_per_side) - patch_jitter

        self.patch_shape = (self.patch_dim, self.patch_dim, number_channels)
        self.patch_count = patches_per_side ** 2

        if self.data_is_3D:
            self.patch_shape = (self.patch_dim,) + self.patch_shape
            self.patch_count = self.patches_per_side ** 3

        self.images_shape = (2,) + self.patch_shape
        self.class_count = self.patch_count - 1

    def apply_model(self):
        if self.data_is_3D:
            self.enc_model, _ = apply_encoder_model_3d(self.patch_shape, **self.kwargs)
        else:
            self.enc_model, _ = apply_encoder_model(self.patch_shape, **self.kwargs)

        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):
        x_input = Input(self.images_shape)
        enc_out = TimeDistributed(encoder_model)(x_input)
        encoder = Model(x_input, enc_out)
        units = np.prod(encoder.outputs[0].shape[1:])
        sub_model = apply_prediction_model((units,), prediction_architecture=self.top_architecture, include_top=False)
        x = Dense(self.class_count, activation="softmax")

        return Sequential([encoder, Flatten(), sub_model, x])

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_training_preprocessing(self):
        def f(x, y):  # not using y here, as it gets generated
            return preprocess_batch(x, self.patches_per_side, self.patch_jitter)

        def f_3d(x, y):
            return preprocess_batch_3d(x, self.patches_per_side, self.patch_jitter)

        if self.data_is_3D:
            return f_3d, f_3d
        else:
            return f, f

    def get_finetuning_model(self, model_checkpoint=None):
        return super(RelativePatchLocationBuilder, self).get_finetuning_model_patches(model_checkpoint)


def create_instance(*params, **kwargs):
    return RelativePatchLocationBuilder(*params, **kwargs)
