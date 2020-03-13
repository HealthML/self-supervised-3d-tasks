from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Flatten, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.pooling import Pooling3D
from tensorflow_core.python.keras.models import load_model

from self_supervised_3d_tasks.custom_preprocessing.jigsaw_preprocess import preprocess_pad
from self_supervised_3d_tasks.custom_preprocessing.relative_patch_location import (
    preprocess_batch,
    preprocess_batch_3d
)
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d

from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model,
    apply_prediction_model_to_encoder, make_finetuning_encoder_3d, make_finetuning_encoder_2d)

class RelativePatchLocationBuilder:
    def __init__(
            self,
            data_dim=384,
            embed_dim=0,  # not using embed dim anymore
            number_channels=3,
            patches_per_side=3,
            patch_jitter=0,
            lr=1e-3,
            train3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        self.cleanup_models = []
        self.data_dim = data_dim
        self.embed_dim = 0
        self.number_channels = number_channels
        self.patch_jitter = patch_jitter
        self.lr = lr
        self.train3D = train3D
        self.kwargs = kwargs
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

        self.enc_model = None
        self.layer_data = None


    def apply_model(self):
        if self.train3D:
            self.enc_model, _ = apply_encoder_model_3d(
                self.patch_shape, self.embed_dim, **self.kwargs
            )
        else:
            self.enc_model = apply_encoder_model(
                self.patch_shape, self.embed_dim, **self.kwargs
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

    def get_finetuning_preprocessing(self):
        def f_identity(x, y):
            return x, y,

        return f_identity, f_identity

    def get_finetuning_model(self, model_checkpoint=None):
        model = self.apply_model()
        assert self.enc_model is not None, "no encoder model"

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        self.cleanup_models.append(model)
        self.cleanup_models.append(self.enc_model)

        if self.train3D:
            model_skips, self.layer_data = make_finetuning_encoder_3d(
                (self.data_dim, self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return model_skips
        else:
            new_enc = make_finetuning_encoder_2d(
                (self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return new_enc

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []

def create_instance(*params, **kwargs):
    return RelativePatchLocationBuilder(*params, **kwargs)
