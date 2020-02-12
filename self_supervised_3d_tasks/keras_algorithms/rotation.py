from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, TimeDistributed
from os.path import expanduser

from self_supervised_3d_tasks.custom_preprocessing.rotation import rotate_batch, resize
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms.custom_utils import apply_encoder_model
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# model_checkpoint = expanduser(
#     "~/workspace/self-supervised-transfer-learning/rotation_ukb_retina/weights-improvement-040.hdf5"
# )


class RotationBuilder:
    def __init__(
            self, data_dim=384, embed_dim=128, n_channels=3, lr=1e-3, **kwargs
    ):
        self.data_dim = data_dim
        self.n_channels = n_channels
        self.lr = lr
        self.data_shape = (data_dim, data_dim)
        self.image_size = data_dim
        self.embed_dim = embed_dim
        self.img_shape = (self.image_size, self.image_size, n_channels)
        self.kwargs = kwargs

    def apply_model(self):
        input_x = Input(self.img_shape)
        enc_model = apply_encoder_model(self.img_shape, self.embed_dim, **self.kwargs)(input_x)
        x = Dense(4, activation='softmax')(enc_model)

        model = Model(inputs=input_x, outputs=x, name="rotation_complete")
        return model

    def get_training_model(self):
        model = self.apply_model()

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
        model = self.apply_model()

        if model_checkpoint is not None:
            model.load_weights(model_checkpoint)

        return model


def create_instance(*params, **kwargs):
    return RotationBuilder(*params, **kwargs)
