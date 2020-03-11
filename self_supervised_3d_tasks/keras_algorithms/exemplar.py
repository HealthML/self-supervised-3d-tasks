import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import Reshape, Dense
from tensorflow.python.keras.layers.pooling import Pooling3D

from self_supervised_3d_tasks.custom_preprocessing.preprocessing_exemplar import (
    preprocessing_exemplar_training,
    get_exemplar_training_preprocessing)
from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_encoder_model_3d,
    apply_encoder_model,
)

from tensorflow.keras.layers import Concatenate, Lambda, Flatten, Input
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model, Sequential


class ExemplarBuilder:
    def __init__(
            self,
            data_dim=384,
            n_channels=3,
            train3D=False,
            alpha_triplet=0.2,
            embed_dim=0,  # not using embed dim anymore
            code_size=1024,
            lr=1e-4,
            model_checkpoint=None,
            sample_neg_examples_from="batch",
            **kwargs
    ):
        """
        init
        :param data_dim: int
        :param n_channels: int
        :param batch_size: int
        :param train3D: bool
        :param alpha_triplet: float for triplet loss
        :param embed_dim: int
        :param lr: float learningrate
        :param model_checkpoint: Dir to model checkpoint
        :param kwargs: ...
        """
        self.sample_neg_examples_from = sample_neg_examples_from
        self.n_channels = n_channels
        self.train3D = train3D
        self.dim = (
            (data_dim, data_dim, data_dim) if self.train3D else (data_dim, data_dim)
        )
        self.alpha_triplet = alpha_triplet
        self.embed_dim = 0
        self.code_size = code_size
        self.lr = lr
        self.model_checkpoint = model_checkpoint
        self.kwargs = kwargs
        self.enc_model = None
        self.cleanup_models = []
        self.layer_data = []

    # TODO: move to losses
    def triplet_loss(self, y_true, y_pred, _alpha=1.0):
        """
        This function returns the calculated triplet loss for y_pred
        :param y_true: not needed
        :param y_pred: predictions of the model
        :param embedding_size: length of embedding
        :param _alpha: defines the shift of the loss
        :return: calculated loss
        """

        positive_distance = K.mean(
            K.square(y_pred[:, 0] - y_pred[:, 1]), axis=-1
        )
        negative_distance = K.mean(
            K.square(y_pred[:, 0] - y_pred[:, 2]), axis=-1
        )
        return K.mean(K.maximum(0.0, positive_distance - negative_distance + _alpha))

    def apply_model(self):
        """
        apply model function to apply the model
        :param input_shape: defines the input shape (dim last)
        :param embed_dim: size of embedding vector
        :return: return the network (encoder) and the compiled model with concated output
        """
        # defines encoder for 3d / non 3d
        if self.train3D:
            # KEEP setting the layer data here, as we will use the enc directly without copying
            self.enc_model, self.layer_data = apply_encoder_model_3d(
                (*self.dim, self.n_channels), self.embed_dim, **self.kwargs
            )
        else:
            self.enc_model = apply_encoder_model(
                (*self.dim, self.n_channels), self.embed_dim, **self.kwargs
            )

        # Define the tensors for the three input images
        input_layer = Input((3, *self.dim, self.n_channels), name="Input")
        anchor_input = Lambda(lambda x: x[:, 0, :], name="anchor_input")(input_layer)
        positive_input = Lambda(lambda x: x[:, 1, :], name="positive_input")(
            input_layer
        )
        negative_input = Lambda(lambda x: x[:, 2, :], name="negative_input")(
            input_layer
        )

        # Generate the encodings (feature vectors) for the three images
        encoded_a = Dense(self.code_size, activation="sigmoid")(Flatten()(self.enc_model(anchor_input)))
        encoded_p = Dense(self.code_size, activation="sigmoid")(Flatten()(self.enc_model(positive_input)))
        encoded_n = Dense(self.code_size, activation="sigmoid")(Flatten()(self.enc_model(negative_input)))

        encoded_a = Reshape((1, self.code_size))(encoded_a)
        encoded_p = Reshape((1, self.code_size))(encoded_p)
        encoded_n = Reshape((1, self.code_size))(encoded_n)

        # Concat the outputs together
        output = Concatenate(axis=-2)([encoded_a, encoded_p, encoded_n])

        # Connect the inputs with the outputs
        model = Model(inputs=input_layer, outputs=output)
        return model

    def get_training_model(self):
        model = self.apply_model()
        # compile with correct triplet loss!
        model.compile(loss=self.triplet_loss, optimizer=Adam(lr=self.lr))
        return model

    def get_training_preprocessing(self):
        f = get_exemplar_training_preprocessing(self.train3D, self.sample_neg_examples_from)
        return f, f

    def get_finetuning_preprocessing(self):
        def f_identity(x, y):
            return x, y,

        return f_identity, f_identity

    def get_finetuning_model(self, model_checkpoint=None):
        org_model = self.apply_model()
        assert self.enc_model is not None, "no encoder model"

        if model_checkpoint is not None:
            org_model.load_weights(model_checkpoint)

        if self.train3D:
            assert self.layer_data is not None, "no layer data for 3D"

            self.layer_data.append(isinstance(self.enc_model.layers[-1], Pooling3D))
            self.cleanup_models.append(self.enc_model)

            self.enc_model = Model(
                inputs=[self.enc_model.layers[0].input],
                outputs=[
                    self.enc_model.layers[-1].output,
                    *reversed(self.layer_data[0]),
                ])

        self.cleanup_models.append(org_model)
        return self.enc_model

    def purge(self):
        for i in reversed(range(len(self.cleanup_models))):
            del self.cleanup_models[i]
        del self.cleanup_models
        self.cleanup_models = []

def create_instance(*params, **kwargs):
    return ExemplarBuilder(*params, **kwargs)
