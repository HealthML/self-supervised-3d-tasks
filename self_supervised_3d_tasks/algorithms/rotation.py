from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model_to_encoder)
from self_supervised_3d_tasks.preprocessing.preprocess_rotation import (
    rotate_batch,
    rotate_batch_3d,
)


class RotationBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            lr=1e-4,
            train3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        super(RotationBuilder, self).__init__(data_dim, number_channels, lr, train3D, **kwargs)

        self.image_size = data_dim
        self.img_shape = (self.image_size, self.image_size, number_channels)
        self.img_shape_3d = (
            self.image_size,
            self.image_size,
            self.image_size,
            number_channels,
        )
        self.top_architecture = top_architecture

    def apply_model(self):
        if self.train3D:
            self.enc_model, self.layer_data = apply_encoder_model_3d(
                self.img_shape_3d, 0, **self.kwargs
            )
            x = Dense(10, activation="softmax")
        else:
            self.enc_model = apply_encoder_model(
                self.img_shape, 0, **self.kwargs
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


    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return RotationBuilder(*params, **kwargs)
