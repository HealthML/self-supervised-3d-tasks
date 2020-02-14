from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.custom_preprocessing.jigsaw_preprocess import (
    preprocess,
    preprocess_resize,
)
from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    load_permutations,
    load_permutations_3d,
)
from self_supervised_3d_tasks.keras_models.fully_connected import fully_connected


class JigsawBuilder:
    def __init__(
            self,
            data_dim=384,
            split_per_side=3,
            patch_jitter=10,
            n_channels=3,
            lr=0.00003,
            embed_dim=128,
            train3D=False,
            **kwargs
    ):
        self.data_dim = data_dim
        self.split_per_side = split_per_side
        self.patch_jitter = patch_jitter
        self.n_channels = n_channels
        self.lr = lr
        self.embed_dim = embed_dim
        self.n_patches = split_per_side * split_per_side
        self.n_patches3D = split_per_side * split_per_side * split_per_side
        self.patch_dim = int((data_dim / split_per_side) - patch_jitter)
        self.train3D = train3D
        self.kwargs = kwargs
        self.cleanup_models = []

    def apply_model(self):
        if self.train3D:
            perms, _ = load_permutations_3d()
        else:
            perms, _ = load_permutations()

        if self.train3D:
            input_x = Input(
                (
                    self.n_patches3D,
                    self.patch_dim,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )
            enc_model = apply_encoder_model_3d(
                (self.patch_dim, self.patch_dim, self.patch_dim, self.n_channels,),
                self.embed_dim, **self.kwargs
            )
        else:
            input_x = Input(
                (self.n_patches, self.patch_dim, self.patch_dim, self.n_channels)
            )
            enc_model = apply_encoder_model(
                (self.patch_dim, self.patch_dim, self.n_channels,), self.embed_dim, **self.kwargs
            )

        x = TimeDistributed(enc_model)(input_x)
        x = Flatten()(x)
        out = fully_connected(x, num_classes=len(perms))

        model = Model(inputs=input_x, outputs=out, name="jigsaw_complete")
        return enc_model, model

    def get_training_model(self):
        model = self.apply_model()[1]
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
            return preprocess(
                x,
                self.split_per_side,
                self.patch_jitter,
                perms,
                is_training=True,
                mode3d=self.train3D,
            )

        def f_val(x, y):
            return preprocess(
                x,
                self.split_per_side,
                self.patch_jitter,
                perms,
                is_training=False,
                mode3d=self.train3D,
            )

        return f_train, f_val

    def get_finetuning_preprocessing(self):
        def f_train(x, y):
            return (
                preprocess_resize(
                    x, self.split_per_side, self.patch_dim, mode3d=self.train3D
                ),
                y,
            )

        def f_val(x, y):
            return (
                preprocess_resize(
                    x, self.split_per_side, self.patch_dim, mode3d=self.train3D
                ),
                y,
            )

        return f_train, f_val

    def get_finetuning_model(self, model_checkpoint=None):
        enc_model, model_full = self.apply_model()

        if model_checkpoint is not None:
            model_full.load_weights(model_checkpoint)

        if self.train3D:
            layer_in = Input(
                (
                    self.n_patches3D,
                    self.patch_dim,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )
        else:
            layer_in = Input(
                (
                    self.n_patches,
                    self.patch_dim,
                    self.patch_dim,
                    self.n_channels,
                )
            )

        layer_out = TimeDistributed(enc_model)(layer_in)
        x = Flatten()(layer_out)

        self.cleanup_models.append(enc_model)
        self.cleanup_models.append(model_full)
        return Model(layer_in, x)

    def purge(self):
        for i in sorted(range(len(self.cleanup_models)), reverse=True):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return JigsawBuilder(*params, **kwargs)
