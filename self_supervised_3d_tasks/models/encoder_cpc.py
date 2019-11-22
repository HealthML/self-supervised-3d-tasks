"""Implements Encder model for Contrastive prediction model.
"""

import tensorflow as tf


def batch_norm(x, training):
    return tf.keras.layers.BatchNormalization(fused=True)(x)


def encoder(x,
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
