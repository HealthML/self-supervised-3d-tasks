from tensorflow.keras import Input, Model
from tensorflow.keras.layers import TimeDistributed, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_jigsaw import (
    preprocess)
from self_supervised_3d_tasks.utils.model_utils import (
    apply_encoder_model,
    apply_encoder_model_3d,
    load_permutations,
    load_permutations_3d,
    apply_prediction_model, make_finetuning_encoder_3d, make_finetuning_encoder_2d)


class JigsawBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            patches_per_side=3,
            patch_jitter=0,
            number_channels=3,
            lr=1e-4,
            data_is_3D=False,
            top_architecture="big_fully",
            **kwargs
    ):
        super(JigsawBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.top_architecture = top_architecture
        self.patches_per_side = patches_per_side
        self.patch_jitter = patch_jitter
        self.n_patches = patches_per_side * patches_per_side
        self.n_patches3D = patches_per_side * patches_per_side * patches_per_side

        self.patch_dim = (data_dim // patches_per_side) - patch_jitter

    def apply_model(self):
        if self.data_is_3D:
            self.enc_model, _ = apply_encoder_model_3d(
                (self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels,), **self.kwargs)
        else:
            self.enc_model, _ = apply_encoder_model((self.patch_dim, self.patch_dim, self.number_channels,),
                                                    **self.kwargs)

        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):
        if self.data_is_3D:
            perms, _ = load_permutations_3d()
            x_input = Input((self.n_patches3D, self.patch_dim, self.patch_dim, self.patch_dim, self.number_channels))
        else:
            perms, _ = load_permutations()
            x_input = Input((self.n_patches, self.patch_dim, self.patch_dim, self.number_channels))

        x = TimeDistributed(self.enc_model)(x_input)
        x = Flatten()(x)

        embedding_layer = apply_prediction_model(x.shape[1:], prediction_architecture=self.top_architecture,
                                                 include_top=False)
        embedding = embedding_layer(x)
        last_layer = Dense(len(perms), activation="softmax")
        out = last_layer(embedding)

        return Model(inputs=x_input, outputs=out, name="jigsaw_complete")

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_training_preprocessing(self):
        if self.data_is_3D:
            perms, _ = load_permutations_3d()
        else:
            perms, _ = load_permutations()

        def f_train(x, y):  # not using y here, as it gets generated
            x, y = preprocess(
                x,
                self.patches_per_side,
                self.patch_jitter,
                perms,
                is_training=True,
                mode3d=self.data_is_3D,
            )
            return x, y

        def f_val(x, y):
            x, y = preprocess(
                x,
                self.patches_per_side,
                self.patch_jitter,
                perms,
                is_training=False,
                mode3d=self.data_is_3D,
            )
            return x, y

        return f_train, f_val

    def get_finetuning_model(self, model_checkpoint=None):
        return super(JigsawBuilder, self).get_finetuning_model_patches(model_checkpoint)

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return JigsawBuilder(*params, **kwargs)
