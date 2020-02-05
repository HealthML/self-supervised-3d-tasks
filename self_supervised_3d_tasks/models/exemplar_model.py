
import os
import numpy as np
from keras.applications import InceptionV3, ResNet50
from keras.optimizers import Adam

np.random.seed(0)
from keras.models import Sequential
from keras.layers import Conv2D, Input, GlobalAveragePooling2D
from keras.models import Model

from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense

from keras.engine.topology import Layer
from keras import backend as K


def build_network(input_shape, embedding_size, model_type="ResNet50"):
    """
    :param input_shape: defines Input Shape (Channel Last)
    :param embedding_size: size of embedding layer
    :param model_type: Resnet50
    :return:
    """
    # define Base Model
    base_model = None
    if model_type =="ResNet50":
        base_model = ResNet50(input_shape=input_shape, include_top=False, weights=None, classes=embedding_size)
    elif model_type == "InceptionV3":
        base_model = InceptionV3(input_shape=input_shape, include_top=False, weights=None, classes=embedding_size)
    return base_model

class TripletLossLayer(Layer):
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


def get_exemplar_model(input_shape, alpha_triplet=0.2, embedding_size=10, lr=0.0006):
    network = build_network(input_shape, embedding_size=embedding_size)
    # Define the tensors for the three input images
    input = Input((3, *input_shape), name="anchor_input")
    anchor_input = Lambda(lambda x: x[0, :])(input)
    positive_input = Lambda(lambda x: x[1, :])(input)
    negative_input = Lambda(lambda x: x[2, :])(input)

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=alpha_triplet, name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])

    optimizer = Adam(lr=lr)

    # Connect the inputs with the outputs
    model = Model(inputs=input, outputs=loss_layer)
    model.compile(loss=None, optimizer=optimizer)
    return model
