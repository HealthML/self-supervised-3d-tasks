
import os
import numpy as np
from keras.optimizers import Adam

np.random.seed(0)
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.models import Model

from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K


def build_network(input_shape, embedding_size):
    '''
    Define the neural network to learn image similarity
    Input : 
        input_shape : shape of input images
        embeddingsize : vectorsize used to encode our picture   
    '''
    # Convolutional Neural Network
    network = Sequential()
    network.add(Conv2D(64, (7, 7), activation='relu',
                   input_shape=input_shape,
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=l2(2e-4)))
    network.add(Flatten())
    network.add(Dense(4096, activation='relu',
                  kernel_regularizer=l2(1e-3),
                  kernel_initializer='he_uniform'))

    network.add(Dense(embedding_size, activation=None,
                  kernel_regularizer=l2(1e-3),
                  kernel_initializer='he_uniform'))

    # Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    print(network.summary())
    return network


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
