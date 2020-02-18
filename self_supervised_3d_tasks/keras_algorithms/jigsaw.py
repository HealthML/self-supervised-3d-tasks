import numpy as np
from tensorflow.keras.layers import Reshape
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, MaxPooling3D, UpSampling3D
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.custom_preprocessing.jigsaw_preprocess import (
    preprocess,
    preprocess_pad,
    preprocess_crop_only)
from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    load_permutations,
    load_permutations_3d,
    apply_prediction_model)
from self_supervised_3d_tasks.keras_models.fully_connected import fully_connected


class JigsawBuilder:
    def __init__(
            self,
            data_dim=384,
            split_per_side=3,
            patch_jitter=0,
            n_channels=3,
            lr=0.00003,
            embed_dim=128,
            train3D=False,
            patch_dim=None,
            top_architecture="big_fully",
            **kwargs
    ):
        self.top_architecture = top_architecture
        self.data_dim = data_dim
        self.split_per_side = split_per_side
        self.patch_jitter = patch_jitter
        self.n_channels = n_channels
        self.lr = lr
        self.embed_dim = embed_dim
        self.n_patches = split_per_side * split_per_side
        self.n_patches3D = split_per_side * split_per_side * split_per_side
        self.patch_dim = int(data_dim / split_per_side)  # we are always padding the jitter away
        self.train3D = train3D
        self.kwargs = kwargs
        self.cleanup_models = []

        self.layer_data = None
        self.enc_model = None

        if patch_dim is not None:
            self.patch_dim = patch_dim

    def apply_model(self):
        if self.train3D:
            perms, _ = load_permutations_3d()

            input_x = Input(
                (
                    self.n_patches3D,
                    self.patch_dim,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )
            self.enc_model, self.layer_data = apply_encoder_model_3d(
                (self.patch_dim, self.patch_dim, self.patch_dim, self.n_channels,),
                self.embed_dim, **self.kwargs
            )
        else:
            perms, _ = load_permutations()

            input_x = Input(
                (self.n_patches, self.patch_dim, self.patch_dim, self.n_channels)
            )
            self.enc_model = apply_encoder_model(
                (self.patch_dim, self.patch_dim, self.n_channels,), self.embed_dim, **self.kwargs
            )

        x = TimeDistributed(self.enc_model)(input_x)
        x = Flatten()(x)

        a = apply_prediction_model(
            x.shape[1:],
            prediction_architecture=self.top_architecture,
            include_top=False,
        )

        last_layer = Dense(len(perms), activation="softmax")

        if a is None:
            out = last_layer(x)
        else:
            out = a(x)
            out = last_layer(out)

        model = Model(inputs=input_x, outputs=out, name="jigsaw_complete")
        return model

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_training_preprocessing(self):
        if self.train3D:
            perms, _ = load_permutations_3d()
        else:
            perms, _ = load_permutations()

        def f_train(x, y):  # not using y here, as it gets generated
            x, y = preprocess(
                x,
                self.split_per_side,
                self.patch_jitter,
                perms,
                is_training=True,
                mode3d=self.train3D,
            )
            return preprocess_pad(x, self.patch_dim, self.train3D), y

        def f_val(x, y):
            x, y = preprocess(
                x,
                self.split_per_side,
                self.patch_jitter,
                perms,
                is_training=False,
                mode3d=self.train3D,
            )
            return preprocess_pad(x, self.patch_dim, self.train3D), y

        return f_train, f_val

    def get_finetuning_preprocessing(self):
        def f(x, y):
            return (
                preprocess_pad(preprocess_crop_only(x, self.split_per_side, False, False), self.patch_dim, False),
                y,
            )

        def f_3d(x, y):
            return (
                preprocess_pad(preprocess_crop_only(x, self.split_per_side, False, True), self.patch_dim, True),
                preprocess_pad(preprocess_crop_only(y, self.split_per_side, False, True), self.patch_dim, True)
            )

        if self.train3D:
            return f_3d, f_3d
        else:
            return f, f

    def get_finetuning_model(self, model_checkpoint=None):
        model_full = self.apply_model()

        if model_checkpoint is not None:
            model_full.load_weights(model_checkpoint)

        if self.train3D:
            assert self.layer_data is not None, "no layer data for 3D"

            layer_in = Input(
                (
                    self.n_patches3D,
                    self.patch_dim,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )

            x = TimeDistributed(self.enc_model)(layer_in)
            flat = Flatten()(x)
            x = Dense(self.n_patches3D * self.embed_dim)(flat)
            x = Reshape((self.n_patches3D, self.embed_dim))(x)

            first_l_shape = self.enc_model.layers[-3].output_shape[1:]
            units = np.prod(first_l_shape)

            model_first_up = Sequential()
            model_first_up.add(Input(self.embed_dim))
            model_first_up.add(Dense(units))
            model_first_up.add(Reshape(first_l_shape))
            if isinstance(self.enc_model.layers[-3], MaxPooling3D):
                model_first_up.add(UpSampling3D((2, 2, 2)))

            print(x.shape)
            x = TimeDistributed(model_first_up)(x)  # we should have correct inputs for the next stage now

            models_skip = [Model(self.enc_model.layers[0].input, x) for x in self.layer_data[0]]
            outputs = [TimeDistributed(m)(layer_in) for m in models_skip]

            result = Model(inputs=[layer_in], outputs=[x, *reversed(outputs)])

        else:
            layer_in = Input(
                (
                    self.n_patches,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )

            layer_out = TimeDistributed(self.enc_model)(layer_in)
            x = Flatten()(layer_out)

            self.cleanup_models.append(self.enc_model)
            self.cleanup_models.append(model_full)

            result = Model(layer_in, x)

        return result

    def purge(self):
        for i in sorted(range(len(self.cleanup_models)), reverse=True):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return JigsawBuilder(*params, **kwargs)
