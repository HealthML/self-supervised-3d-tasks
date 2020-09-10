import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, TimeDistributed

from self_supervised_3d_tasks.algorithms.algorithm_base import AlgorithmBuilderBase
from self_supervised_3d_tasks.preprocessing.preprocess_cpc import (
    preprocess_grid_2d,
    preprocess_3d,
    preprocess_grid_3d,
    preprocess_2d
)
from self_supervised_3d_tasks.utils.model_utils import apply_encoder_model_3d, apply_encoder_model


def network_autoregressive(x):
    x = keras.layers.GRU(units=256, return_sequences=False)(x)
    return x


def network_prediction(context, code_size, predict_terms):
    outputs = []
    for i in range(predict_terms):
        outputs.append(
            keras.layers.Dense(units=code_size, activation="linear")(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1


class CPCBuilder(AlgorithmBuilderBase):
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            crop_size=None,
            patches_per_side=7,
            code_size=1024,
            lr=1e-3,
            data_is_3D=False,
            **kwargs,
    ):
        super(CPCBuilder, self).__init__(data_dim, number_channels, lr, data_is_3D, **kwargs)

        if crop_size is None:
            crop_size = int(data_dim * 0.95)

        self.crop_size = crop_size
        self.patches_per_side = patches_per_side
        self.code_size = code_size

        # run a test to obtain data sizes
        prep_train = self.get_training_preprocessing()[0]
        test_data = np.zeros((1, data_dim, data_dim, data_dim, number_channels), dtype=np.float32) if self.data_is_3D \
            else np.zeros((1, data_dim, data_dim, number_channels), dtype=np.float32)

        test_x = prep_train(test_data, test_data)[0]
        self.terms = test_x[0].shape[1]
        self.image_size = test_x[0].shape[2]
        self.predict_terms = test_x[1].shape[1]

        self.img_shape = (self.image_size, self.image_size, self.number_channels)
        self.img_shape_3d = (self.image_size, self.image_size, self.image_size, self.number_channels)

    def apply_model(self):
        if self.data_is_3D:
            self.enc_model, _ = apply_encoder_model_3d(self.img_shape_3d, **self.kwargs)
        else:
            self.enc_model, _ = apply_encoder_model(self.img_shape, **self.kwargs)

        return self.apply_prediction_model_to_encoder(self.enc_model)

    def apply_prediction_model_to_encoder(self, encoder_model):
        if self.data_is_3D:
            x_input = Input((self.terms, self.image_size, self.image_size, self.image_size, self.number_channels))
            y_input = Input(
                (self.predict_terms, self.image_size, self.image_size, self.image_size, self.number_channels))
        else:
            x_input = Input((self.terms, self.image_size, self.image_size, self.number_channels))
            y_input = Input((self.predict_terms, self.image_size, self.image_size, self.number_channels))
        model_with_embed_dim = Sequential([encoder_model, Flatten(), Dense(self.code_size)])
        x_encoded = TimeDistributed(model_with_embed_dim)(x_input)
        context = network_autoregressive(x_encoded)
        preds = network_prediction(context, self.code_size, self.predict_terms)

        y_encoded = TimeDistributed(model_with_embed_dim)(y_input)
        dot_product_probs = CPCLayer()([preds, y_encoded])
        cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

        return cpc_model

    def get_training_model(self):
        model = self.apply_model()
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
        )

        return model

    def get_training_preprocessing(self):
        def f(x, y):  # not using y here, as it gets generated
            return preprocess_grid_2d(preprocess_2d(x, self.crop_size, self.patches_per_side))

        def f_3d(x, y):  # not using y here, as it gets generated
            return preprocess_grid_3d(preprocess_3d(x, self.crop_size, self.patches_per_side))

        if self.data_is_3D:
            return f_3d, f_3d
        else:
            return f, f

    def get_finetuning_model(self, model_checkpoint=None):
        return super(CPCBuilder, self).get_finetuning_model_patches(model_checkpoint)


def create_instance(*params, **kwargs):
    return CPCBuilder(*params, **kwargs)
