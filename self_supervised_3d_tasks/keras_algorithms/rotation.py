from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.custom_preprocessing.rotation_preprocess import rotate_batch, rotate_batch_3d
from self_supervised_3d_tasks.keras_algorithms.custom_utils import apply_encoder_model, apply_encoder_model_3d


class RotationBuilder:
    def __init__(
            self, data_dim=384, embed_dim=1024, n_channels=3, lr=1e-3, train3D=False, **kwargs
    ):
        self.data_dim = data_dim
        self.n_channels = n_channels
        self.lr = lr
        self.image_size = data_dim
        self.embed_dim = embed_dim
        self.img_shape = (self.image_size, self.image_size, n_channels)
        self.img_shape_3d = (self.image_size, self.image_size, self.image_size, n_channels)
        self.kwargs = kwargs
        self.cleanup_models = []
        self.train3D = train3D

    def apply_model(self):
        if self.train3D:
            enc_model = apply_encoder_model_3d(self.img_shape_3d, self.embed_dim, **self.kwargs)
            x = Dense(10, activation='softmax')
        else:
            enc_model = apply_encoder_model(self.img_shape, self.embed_dim, **self.kwargs)
            x = Dense(4, activation='softmax')

        model = Sequential([enc_model, x])
        enc_model.summary()
        model.summary()

        return model, enc_model

    def get_training_model(self):
        model, _ = self.apply_model()

        model.compile(
            optimizer=Adam(lr=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
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
        org_model, enc_model = self.apply_model()

        if model_checkpoint is not None:
            org_model.load_weights(model_checkpoint)

        self.cleanup_models.append(org_model)
        self.cleanup_models.append(enc_model)
        return enc_model

    def purge(self):
        for i in sorted(range(len(self.cleanup_models)), reverse=True):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return RotationBuilder(*params, **kwargs)
