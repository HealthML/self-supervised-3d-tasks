from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Flatten, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.pooling import Pooling3D

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
            patch_dim=None,
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
        self.patch_dim = int(data_dim / patches_per_side)

        if patch_dim is not None:
            self.patch_dim = patch_dim

        self.patch_shape = (self.patch_dim, self.patch_dim, n_channels)
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
            self.enc_model, self.layer_data = apply_encoder_model_3d(
                self.patch_shape, self.embed_dim, **self.kwargs
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
                self.patch_shape, self.embed_dim, **self.kwargs
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
            x,y = preprocess_batch(x, self.patches_per_side, self.patch_jitter)
            return preprocess_pad(x, self.patch_dim, False), y

        def f_3d(x, y):
            x,y = preprocess_batch_3d(x, self.patches_per_side, self.patch_jitter)
            return preprocess_pad(x, self.patch_dim, True), y

        if self.train3D:
            return f_3d, f_3d

        return f, f

    def get_finetuning_preprocessing(self):
        def f(x, y):  # not using y here, as it gets generated
            return (
                preprocess_pad(preprocess_batch(x, self.patches_per_side, 0, is_training=False)[0],
                               self.patch_dim, False),
                               y)

        def f_3d(x, y):
            x = preprocess_batch_3d(x, self.patches_per_side, 0, is_training=False)[0]
            y = preprocess_batch_3d(y, self.patches_per_side, 0, is_training=False)[0]
            return preprocess_pad(x, self.patch_dim, True), preprocess_pad(y, self.patch_dim, True)

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

            layer_in = Input(
                (
                    self.patch_count,
                    self.patch_dim,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )
            out_one = TimeDistributed(self.enc_model)(layer_in)
            models_skip = [Model(self.enc_model.layers[0].input, x) for x in self.layer_data[0]]
            outputs = [TimeDistributed(m)(layer_in) for m in models_skip]

            result = Model(inputs=[layer_in], outputs=[out_one, *reversed(outputs)])

            self.layer_data.append((self.enc_model.layers[-3].output_shape[1:],
                                    isinstance(self.enc_model.layers[-3], Pooling3D)))

            self.cleanup_models += [*models_skip, result]
            self.cleanup_models.append(model)

            return result
        else:
            layer_in = Input(
                (
                    self.patch_count,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )

            layer_out = TimeDistributed(self.enc_model)(layer_in)
            x = Flatten()(layer_out)

            self.cleanup_models.append(self.enc_model)
            self.cleanup_models.append(model)

            return Model(layer_in, x)


    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []

def create_instance(*params, **kwargs):
    return RelativePatchLocationBuilder(*params, **kwargs)
