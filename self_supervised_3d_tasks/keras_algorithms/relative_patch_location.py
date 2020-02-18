from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Flatten, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.custom_preprocessing.relative_patch_location import (
    preprocess_batch,
    resize,
    preprocess_batch_3d
)
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d

from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    apply_prediction_model,
)

class RelativePatchLocationBuilder:
    def __init__(
            self,
            data_dim=384,
            embed_dim=1024,
            n_channels=3,
            patches_per_side=3,
            patch_jitter=0,
            lr=1e-3,
            train3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        self.data_dim = data_dim
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.patches_per_side = patches_per_side
        self.patch_jitter = patch_jitter
        self.lr = lr
        self.image_size = int(data_dim / patches_per_side) - patch_jitter
        self.img_shape = (self.image_size, self.image_size, n_channels)
        self.img_shape_3d = (self.image_size, self.image_size, self.image_size, n_channels)
        self.train3D = train3D
        self.kwargs = kwargs
        self.top_architecture = top_architecture

    def apply_model(self):
        if self.train3D:
            enc_model = apply_encoder_model_3d(
                self.img_shape_3d, self.embed_dim, **self.kwargs
            )
            a = apply_prediction_model(
                self.embed_dim,
                prediction_architecture=self.top_architecture,
                include_top=False,
            )
            x = Dense(27, activation="softmax")
        else:
            enc_model = apply_encoder_model(
                self.img_shape, self.embed_dim, **self.kwargs
            )
            a = apply_prediction_model(
                self.embed_dim,
                prediction_architecture=self.top_architecture,
                include_top=False,
            )
            # TODO Changed 4 => 9 (classes)
            x = Dense(9, activation="softmax")

        model = Sequential([enc_model, a, x])
        enc_model.summary()
        model.summary()

        return model, enc_model

    def get_training_model(self):
        model, _ = self.apply_model()

        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
    #
    # def get_training_model(self):
    #     model = get_res_net_2d(
    #         input_shape=self.img_shape,
    #         classes=self.patches_per_side ** 2,
    #         architecture="ResNet50",
    #         learning_rate=self.lr,
    #     )
    #
    #     return model

    def get_training_preprocessing(self):
        def f(x, y):  # not using y here, as it gets generated
            return preprocess_batch(x, self.patches_per_side, self.patch_jitter)

        def f_3d(x, y):
            return preprocess_batch_3d(x, self.patches_per_side, self.patch_jitter)

        if self.train3D:
            return f_3d, f_3d

        return f, f

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

    def get_finetuning_model(self, model_checkpoint=None):
        model = get_res_net_2d(
            input_shape=[self.image_size, self.image_size, self.n_channels],
            classes=self.patches_per_side ** 2,
            architecture="ResNet50",
            learning_rate=self.lr,
        )

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        layer_in = Input((self.patches_per_side * self.patches_per_side,) + self.img_shape)
        layer_out = TimeDistributed(model)(layer_in)

        x = Flatten()(layer_out)

        return Model(layer_in, x), [model]


def create_instance(*params, **kwargs):
    return RelativePatchLocationBuilder(*params, **kwargs)
