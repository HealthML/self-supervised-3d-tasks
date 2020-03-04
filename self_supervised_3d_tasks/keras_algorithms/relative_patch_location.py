from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Flatten, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.pooling import Pooling3D

from self_supervised_3d_tasks.custom_preprocessing.relative_patch_location import (
    preprocess_batch,
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
        self.cleanup_models = []
        self.data_dim = data_dim
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.patch_jitter = patch_jitter
        self.lr = lr
        self.train3D = train3D
        self.kwargs = kwargs
        self.top_architecture = top_architecture

        self.patches_per_side = patches_per_side
        self.image_size = int(data_dim / patches_per_side)
        self.image_shape = (self.image_size, self.image_size, n_channels)
        self.patch_count = patches_per_side**2
        if self.train3D:
            self.image_shape = (self.image_size, ) + self.image_shape
            self.patch_count = self.patches_per_side**3

        self.images_shape = (2, ) + self.image_shape
        self.class_count = self.patch_count - 1

        self.enc_model = None
        self.layer_data = None


    def apply_model(self):
        if self.train3D:
            self.enc_model, self.layer_data = apply_encoder_model_3d(
                self.image_shape, self.embed_dim, **self.kwargs
            )
            self.enc_model.summary()
            a = apply_prediction_model(
                2 * self.embed_dim,
                prediction_architecture=self.top_architecture,
                include_top=False,
            )
            a.summary()
        else:
            self.enc_model = apply_encoder_model(
                self.image_shape, self.embed_dim, **self.kwargs
            )
            self.enc_model.summary()
            a = apply_prediction_model(
                2 * self.embed_dim,
                prediction_architecture=self.top_architecture,
                include_top=False,
            )
            a.summary()

        x_input = Input(self.images_shape)
        enc_out = TimeDistributed(self.enc_model)(x_input)
        enc_out = Flatten()(enc_out)
        enc_out = a(enc_out)
        out = Dense(self.class_count, activation="softmax")(enc_out)

        model = Model(x_input, out)
        model.summary()

        return model, self.enc_model

    def get_training_model(self):
        model, _ = self.apply_model()

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

        return f, f

    def get_finetuning_preprocessing(self):
        def f(x, y):  # not using y here, as it gets generated
            return preprocess_batch(x, self.patches_per_side, 0, is_training=False)[0], y

        def f_3d(x, y):
            return preprocess_batch_3d(x, self.patches_per_side, 0, is_training=False)[0], y

        if self.train3D:
            return f_3d, f_3d

        return f, f

    def get_finetuning_model(self, model_checkpoint=None):
        model, enc_model = self.apply_model()

        self.cleanup_models.append(model)
        self.cleanup_models.append(enc_model)

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        #####
        if self.train3D:
            assert self.layer_data is not None, "no layer data for 3D"

            self.layer_data.append((self.enc_model.layers[-3].output_shape[1:],
                                    isinstance(self.enc_model.layers[-3], Pooling3D)))

            # self.cleanup_models.append(self.enc_model)
            self.enc_model = Model(inputs=[self.enc_model.layers[0].input],
                                   outputs=[self.enc_model.layers[-1].output,
                                   *reversed(self.layer_data[0])])
        #####

        layer_in = Input((self.patch_count,) + self.image_shape)
        layer_out = TimeDistributed(enc_model)(layer_in)

        x = Flatten()(layer_out)

        return Model(layer_in, x)

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []

def create_instance(*params, **kwargs):
    return RelativePatchLocationBuilder(*params, **kwargs)
