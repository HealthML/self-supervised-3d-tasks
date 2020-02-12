from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.custom_preprocessing.rotation import rotate_batch
from self_supervised_3d_tasks.keras_algorithms.custom_utils import apply_encoder_model


class RotationBuilder:
    def __init__(
            self, data_dim=384, embed_dim=1024, n_channels=3, lr=1e-3, **kwargs
    ):
        self.data_dim = data_dim
        self.n_channels = n_channels
        self.lr = lr
        self.data_shape = (data_dim, data_dim)
        self.image_size = data_dim
        self.embed_dim = embed_dim
        self.img_shape = (self.image_size, self.image_size, n_channels)
        self.kwargs = kwargs
        self.cleanup_models = []

    def apply_model(self):
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
        def f_train(x, y):  # not using y here, as it gets generated
            return rotate_batch(x, y)

        def f_val(x, y):
            return rotate_batch(x, y)

        return f_train, f_val

    def get_finetuning_preprocessing(self):
        def f_train(x, y):
            return x, y

        def f_val(x, y):
            return x, y

        return f_train, f_val

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
