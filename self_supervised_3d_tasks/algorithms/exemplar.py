from tensorflow.keras.layers import Concatenate, Lambda, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Reshape, Dense

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.utils.model_utils import (
    apply_encoder_model_3d,
    apply_encoder_model,
)
from self_supervised_3d_tasks.utils.metrics import triplet_loss
from self_supervised_3d_tasks.preprocessing.preprocess_exemplar import (
    get_exemplar_training_preprocessing)


class ExemplarBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            data_is_3D=False,
            code_size=1024,
            lr=1e-4,
            sample_neg_examples_from="batch",
            **kwargs
    ):
        super(ExemplarBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        self.sample_neg_examples_from = sample_neg_examples_from
        self.dim = (
            (data_dim, data_dim, data_dim) if self.data_is_3D else (data_dim, data_dim)
        )
        self.code_size = code_size

    def apply_model(self):
        if self.data_is_3D:
            self.enc_model, self.layer_data = apply_encoder_model_3d((*self.dim, self.number_channels), **self.kwargs)
        else:
            self.enc_model, self.layer_data = apply_encoder_model((*self.dim, self.number_channels), **self.kwargs)
        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):
        input_layer = Input((3, *self.dim, self.number_channels), name="Input")
        anchor_input = Lambda(lambda x: x[:, 0, :], name="anchor_input")(input_layer)
        positive_input = Lambda(lambda x: x[:, 1, :], name="positive_input")(input_layer)
        negative_input = Lambda(lambda x: x[:, 2, :], name="negative_input")(input_layer)

        encoded_a = Dense(self.code_size, activation="sigmoid")(Flatten()(self.enc_model(anchor_input)))
        encoded_p = Dense(self.code_size, activation="sigmoid")(Flatten()(self.enc_model(positive_input)))
        encoded_n = Dense(self.code_size, activation="sigmoid")(Flatten()(self.enc_model(negative_input)))
        encoded_a = Reshape((1, self.code_size))(encoded_a)
        encoded_p = Reshape((1, self.code_size))(encoded_p)
        encoded_n = Reshape((1, self.code_size))(encoded_n)

        output = Concatenate(axis=-2)([encoded_a, encoded_p, encoded_n])

        return Model(inputs=input_layer, outputs=output)

    def get_training_model(self):
        model = self.apply_model()
        model.compile(loss=triplet_loss, optimizer=Adam(lr=self.lr))
        return model

    def get_training_preprocessing(self):
        f = get_exemplar_training_preprocessing(self.data_is_3D, self.sample_neg_examples_from)
        return f, f


def create_instance(*params, **kwargs):
    return ExemplarBuilder(*params, **kwargs)
