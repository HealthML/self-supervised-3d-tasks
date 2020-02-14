from os.path import expanduser

import numpy as np
from functools import partial
from self_supervised_3d_tasks.custom_preprocessing.preprocessing_exemplar import preprocessing_exemplar

from self_supervised_3d_tasks.algorithms import patch_utils
from keras.applications import ResNet50
from keras.optimizers import Adam

np.random.seed(0)

from keras.layers import Input, TimeDistributed, Concatenate
from keras.models import Model

from keras.layers.core import Lambda, Flatten, Dense

from keras.engine.topology import Layer
from keras import backend as K


number_channels = 3
dim = (384, 384)
batch_size = 10
n_channels = number_channels
dim_3d = False
model_params = {
            "alpha_triplet": 0.2,
            "embedding_size": 10,
            "lr": 0.0006
            }

# TODO adjust model checkpiont
model_checkpoint = ""


class TripletLossLayer(Layer):
    """
    I am a layer for the loss function. I am not used anymore because I am not compatible with model.fit
    """
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def triplet_loss(y_true, y_pred, embedding_size=10, _alpha=.5):
    """
    This function returns the calculated triplet loss for y_pred
    :param y_true: not needed
    :param y_pred: predictions of the model
    :param embedding_size: length of embedding
    :param _alpha: defines the shift of the loss
    :return: calculated loss
    """
    embeddings = K.reshape(y_pred, (-1, 3, embedding_size))

    positive_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,1]),axis=-1)
    negative_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,2]),axis=-1)
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + _alpha))


def apply_model(input_shape, embedding_size=10, lr=0.0006):
    """
    apply model function to apply the model
    :param input_shape: defines the input shape (dim last)
    :param embedding_size: number of embedded layers
    :param lr: learning rate
    :return: return the network (encoder) and the compiled model with concated output
    """
    # defines encoder
    network = ResNet50(input_shape=input_shape, include_top=True, weights=None, classes=embedding_size)
    # Define the tensors for the three input images
    input = Input((3, *input_shape), name="anchor_input")
    anchor_input = Lambda(lambda x: x[0, :])(input)
    positive_input = Lambda(lambda x: x[1, :])(input)
    negative_input = Lambda(lambda x: x[2, :])(input)

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    # Concat the outputs together
    output = Concatenate()([encoded_a, encoded_p, encoded_n])

    # set optimizer
    optimizer = Adam(lr=lr)

    # Connect the inputs with the outputs
    model = Model(inputs=input, outputs=output)
    # compile the model
    model.compile(loss=triplet_loss, optimizer=optimizer)
    return network, model


def get_training_model():
    return apply_model((*dim, number_channels))[1]


def get_training_preprocessing():
    perms, _ = patch_utils.load_permutations()

    def f_train(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    def f_val(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    return f_train, f_val


def get_finetuning_preprocessing():
    def f_train(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    def f_val(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    return f_train, f_val


def get_finetuning_layers(load_weights, freeze_weights):
    enc_model, model_full = apply_model((*dim, number_channels))

    if load_weights:
        model_full.load_weights(model_checkpoint)

    if freeze_weights:
        # freeze the encoder weights
        enc_model.trainable = False

    layer_in = Input((*dim, number_channels), name="anchor_input")
    layer_out = TimeDistributed(enc_model)(layer_in)

    x = Flatten()(layer_out)
    return layer_in, x, [enc_model, model_full]