import numpy as np
from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import load_model, Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import UpSampling3D
from tensorflow.python.keras.layers.pooling import Pooling3D

from self_supervised_3d_tasks.custom_preprocessing.rotation_preprocess import (
    rotate_batch,
    rotate_batch_3d,
)
from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model,
    flatten_model, print_flat_summary, apply_prediction_model_to_encoder)


class RotationBuilder:
    def __init__(
            self,
            data_dim=384,
            embed_dim=0,  # not using embed dim anymore
            n_channels=3,
            lr=1e-4,
            train3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        self.data_dim = data_dim
        self.n_channels = n_channels
        self.lr = lr
        self.image_size = data_dim
        self.embed_dim = 0
        self.img_shape = (self.image_size, self.image_size, n_channels)
        self.img_shape_3d = (
            self.image_size,
            self.image_size,
            self.image_size,
            n_channels,
        )
        self.top_architecture = top_architecture
        self.train3D = train3D

        self.kwargs = kwargs
        self.enc_model = None
        self.cleanup_models = []
        self.layer_data = []

    def apply_model(self):
        if self.train3D:
            self.enc_model, self.layer_data = apply_encoder_model_3d(
                self.img_shape_3d, self.embed_dim, **self.kwargs
            )
            x = Dense(10, activation="softmax")
        else:
            self.enc_model = apply_encoder_model(
                self.img_shape, self.embed_dim, **self.kwargs
            )
            x = Dense(4, activation="softmax")

        return apply_prediction_model_to_encoder(
            self.enc_model,
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
            return rotate_batch(x, y)

        def f_3d(x, y):
            return rotate_batch_3d(x, y)

        if self.train3D:
            return f_3d, f_3d
        else:
            return f, f

    def get_finetuning_preprocessing(self):
        def f_identity(x, y):
            return x, y

        return f_identity, f_identity

    def get_finetuning_model(self, model_checkpoint=None):
        org_model = self.apply_model()
        assert self.enc_model is not None, "no encoder model"

        if model_checkpoint is not None:
            org_model.load_weights(model_checkpoint)

        if self.train3D:
            assert self.layer_data is not None, "no layer data for 3D"

            self.layer_data.append(isinstance(self.enc_model.layers[-1], Pooling3D))
            self.cleanup_models.append(self.enc_model)

            self.enc_model = Model(
                inputs=[self.enc_model.layers[0].input],
                outputs=[
                    self.enc_model.layers[-1].output,
                    *reversed(self.layer_data[0])
                ])

        self.cleanup_models.append(org_model)
        return self.enc_model

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return RotationBuilder(*params, **kwargs)
