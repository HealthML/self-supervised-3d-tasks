from tensorflow.keras import Model, Input
from tensorflow.keras.layers import TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_rpl import (
    preprocess_batch,
    preprocess_batch_3d
)

from self_supervised_3d_tasks.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model_to_encoder, make_finetuning_encoder_3d, make_finetuning_encoder_2d)

class RelativePatchLocationBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            patches_per_side=3,
            patch_jitter=0,
            lr=1e-3,
            train3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        super(RelativePatchLocationBuilder, self).__init__(data_dim, number_channels, lr, train3D, **kwargs)

        self.patch_jitter = patch_jitter
        self.top_architecture = top_architecture

        self.patches_per_side = patches_per_side
        self.patch_dim = (data_dim // patches_per_side) - patch_jitter

        self.patch_shape = (self.patch_dim, self.patch_dim, number_channels)
        self.patch_count = patches_per_side**2

        if self.train3D:
            self.patch_shape = (self.patch_dim,) + self.patch_shape
            self.patch_count = self.patches_per_side**3

        self.images_shape = (2, ) + self.patch_shape
        self.class_count = self.patch_count - 1


    def apply_model(self):
        if self.train3D:
            self.enc_model, _ = apply_encoder_model_3d(
                self.patch_shape, **self.kwargs
            )
        else:
            self.enc_model = apply_encoder_model(
                self.patch_shape, **self.kwargs
            )

        x_input = Input(self.images_shape)
        enc_out = TimeDistributed(self.enc_model)(x_input)

        x = Dense(self.class_count, activation="softmax")
        return apply_prediction_model_to_encoder(
            Model(x_input, enc_out),
            prediction_architecture=self.top_architecture,
            include_top=False,
            model_on_top=x
        )

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

        if self.train3D:
            return f_3d, f_3d
        else:
            return f, f

    def get_finetuning_model(self, model_checkpoint=None):
        return super(RelativePatchLocationBuilder, self).get_finetuning_model_patches(model_checkpoint)


def create_instance(*params, **kwargs):
    return RelativePatchLocationBuilder(*params, **kwargs)
