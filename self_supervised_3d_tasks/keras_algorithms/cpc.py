import numpy as np

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, TimeDistributed
from tensorflow.python.keras.layers.pooling import Pooling3D

from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import (
    preprocess_grid,
    preprocess
)
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess_3d import preprocess_3d, preprocess_grid_3d
from self_supervised_3d_tasks.keras_algorithms.custom_utils import apply_encoder_model_3d, apply_encoder_model, \
    make_finetuning_encoder_3d, make_finetuning_encoder_2d


def network_autoregressive(x):
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)
    return x


def network_prediction(context, code_size, predict_terms):
    outputs = []
    for i in range(predict_terms):
        outputs.append(
            keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


class CPCBuilder:
    def __init__(
            self,
            data_dim=384,
            number_channels=3,
            crop_size=None,
            patches_per_side=7,
            embed_dim=0,  # not using embed dim anymore
            code_size=1024,
            lr=1e-3,
            train3D=False,
            **kwargs,
    ):
        if crop_size is None:
            crop_size = int(data_dim * 0.95)

        self.data_dim = data_dim
        self.number_channels = number_channels
        self.crop_size = crop_size
        self.patches_per_side = patches_per_side
        self.code_size = code_size
        self.lr = lr
        self.kwargs = kwargs
        self.train3D = train3D

        # run a test to obtain data sizes
        prep_train = self.get_training_preprocessing()[0]
        test_data = np.zeros((1, data_dim, data_dim, data_dim, number_channels), dtype=np.float32) if self.train3D \
            else np.zeros((1, data_dim, data_dim, number_channels), dtype=np.float32)

        test_x = prep_train(test_data, test_data)[0]
        self.terms = test_x[0].shape[1]
        self.image_size = test_x[0].shape[2]
        self.predict_terms = test_x[1].shape[1]

        self.img_shape = (self.image_size, self.image_size, self.number_channels)
        self.img_shape_3d = (self.image_size, self.image_size, self.image_size, self.number_channels)

        self.cleanup_models = []
        self.enc_model = None
        self.layer_data = None

    def apply_model(self):
        if self.train3D:
            self.enc_model, _ = apply_encoder_model_3d(self.img_shape_3d, 0, **self.kwargs)
            x_input = Input((self.terms, self.image_size, self.image_size, self.image_size, self.number_channels))
            y_input = keras.layers.Input(
                (self.predict_terms, self.image_size, self.image_size, self.image_size, self.number_channels))
        else:
            self.enc_model = apply_encoder_model(self.img_shape, 0, **self.kwargs)
            x_input = Input((self.terms, self.image_size, self.image_size, self.number_channels))
            y_input = keras.layers.Input((self.predict_terms, self.image_size, self.image_size, self.number_channels))

        # CPC really needs a code size
        model_with_embed_dim = Sequential([self.enc_model, Flatten(), Dense(self.code_size)])
        x_encoded = TimeDistributed(model_with_embed_dim)(x_input)
        context = network_autoregressive(x_encoded)
        preds = network_prediction(context, self.code_size, self.predict_terms)

        y_encoded = keras.layers.TimeDistributed(model_with_embed_dim)(y_input)
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
            return preprocess_grid(preprocess(x, self.crop_size, self.patches_per_side))

        def f_3d(x, y):  # not using y here, as it gets generated
            return preprocess_grid_3d(preprocess_3d(x, self.crop_size, self.patches_per_side))

        if self.train3D:
            return f_3d, f_3d
        else:
            return f, f

    def get_finetuning_preprocessing(self):
        def f_identity(x, y):
            return x, y

        return f_identity, f_identity

    def get_finetuning_model(self, model_checkpoint=None):
        cpc_model = self.apply_model()

        if model_checkpoint is not None:
            cpc_model.load_weights(model_checkpoint)

        self.cleanup_models.append(self.enc_model)
        self.cleanup_models.append(cpc_model)

        if self.train3D:
            model_skips, self.layer_data = make_finetuning_encoder_3d(
                (self.data_dim, self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return model_skips
        else:
            new_enc = make_finetuning_encoder_2d(
                (self.data_dim, self.data_dim, self.number_channels,),
                self.enc_model,
                **self.kwargs
            )

            return new_enc

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []


def create_instance(*params, **kwargs):
    return CPCBuilder(*params, **kwargs)
