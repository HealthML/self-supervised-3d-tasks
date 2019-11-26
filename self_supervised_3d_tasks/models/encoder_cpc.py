"""Implements Encder model for Contrastive prediction model.
"""

import tensorflow as tf
import tensorflow.keras as keras


def batch_norm(x, training):
    return tf.keras.layers.BatchNormalization(fused=True)(x)


def encoder(x,
            is_training: bool,
            num_layers: int = 3,
            # TODO: extract to config (3D or 2D)
            strides=(2, 2),
            code_size: int = 128,
            filters: int = 4,
            weight_decay: float = 1e-4,
            kernel_size: int = 7,
            activation_fn=tf.nn.relu,
            normalization_fn=batch_norm,
            num_classes: int = None,
            **params):

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x

def encoder_x(x,
            is_training: bool,
            num_layers: int = 3,
            # TODO: extract to config (3D or 2D)
            strides=(2, 2),
            outputs: int = 1000,
            filters: int = 4,
            weight_decay: float = 1e-4,
            kernel_size: int = 7,
            activation_fn=tf.nn.relu,
            normalization_fn=batch_norm,
            num_classes: int = None
            ):
    '''
    Args:
        x: Input layer
        is_training (bool): if the model is training
        num_layers (int): number of convolution layer
        strides (tuple): strides for convolution layer
        outputs (int): number of outputs
        filters (int): number of filters in convolution layer
        weight_decay (float): scale for kernel regularization
        kernel_size: size of kernel in convolutions
        activation_fn: activation function of dense layer
        normalization_fn: normalization function after convolution layers

    Returns:
        x: last dense layer
    '''

    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    for i in range(1, num_layers):
        x = tf.keras.layers.Conv2D(filters=4,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='VALID', use_bias=False,
                                   kernel_regularizer=kernel_regularizer)(x)

        x = normalization_fn(x, training=is_training)

    x = tf.keras.layers.Dense(units=outputs,
                              activation=activation_fn,
                              )(x)

    return x
