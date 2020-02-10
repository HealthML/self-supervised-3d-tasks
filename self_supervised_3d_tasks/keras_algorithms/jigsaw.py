from os.path import expanduser

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


# data_dir="/mnt/mpws2019cl1/retinal_fundus/left/max_256/"
# model_checkpoint = expanduser(
#     "~/workspace/self-supervised-transfer-learning/jigsaw_kaggle_retina_3/weights-improvement-059.hdf5"
# )


class JigsawBuilder:
    def __init__(
            self,
            h_w=384,
            split_per_side=3,
            patch_jitter=10,
            n_channels=3,
            lr=0.00003,
            embed_dim=1000,
            architecture="ResNet50",
            train3D=False,
            **kwargs
    ):
        self.h_w = h_w
        self.split_per_side = split_per_side
        self.patch_jitter = patch_jitter
        self.n_channels = n_channels
        self.lr = lr
        self.embed_dim = embed_dim
        self.architecture = architecture
        self.n_patches = split_per_side * split_per_side
        self.n_patches3D = split_per_side * split_per_side * split_per_side
        self.dim = (h_w, h_w)
        self.dim3D = (h_w, h_w, h_w)
        self.patch_dim = int((h_w / split_per_side) - patch_jitter)
        self.train3D = train3D

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
                self.embed_dim,
            )
        else:
            input_x = Input(
                (self.n_patches, self.patch_dim, self.patch_dim, self.n_channels)
            )
            enc_model = apply_encoder_model(
                (self.patch_dim, self.patch_dim, self.n_channels,), self.embed_dim
            )

        x = TimeDistributed(enc_model)(input_x)
        x = Flatten()(x)
        out = fully_connected(x, num_classes=len(perms))

        model = Model(inputs=input_x, outputs=out)
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        return enc_model, model

    def get_training_model(self):
        return self.apply_model()[1]

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

        layer_in = Input(
            (
                self.split_per_side * self.split_per_side,
                self.patch_dim,
                self.patch_dim,
                self.n_channels,
            )
        )
        layer_out = TimeDistributed(enc_model)(layer_in)

        x = Flatten()(layer_out)
        return Model(layer_in, x), [enc_model, model_full]


def create_instance(*params, **kwargs):
    return JigsawBuilder(*params, **kwargs)
