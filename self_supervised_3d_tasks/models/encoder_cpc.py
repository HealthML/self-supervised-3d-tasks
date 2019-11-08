"""Implements Encder model for Contrastive prediction model.
"""

import functools

import tensorflow as tf


def batch_norm(x, training):
    return tf.layers.batch_normalization(x, fused=True, training=training)


def encoder(x,  # pylint: disable=missing-docstring
            is_training,
            num_layers,
            strides=(2, 2, 2),
            num_classes=1000,
            filters_factor=4,
            weight_decay=1e-4,
            include_root_block=True,
            root_conv_size=7, root_conv_stride=2,
            root_pool_size=3, root_pool_stride=2,
            activation_fn=tf.nn.relu,
            last_relu=True,
            normalization_fn=batch_norm,
            global_pool=True):
    end_points = {}

    filters = 16 * filters_factor

    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    x = tf.layers.conv2d(x, filters=filters,
                         kernel_size=root_conv_size,
                         strides=root_conv_stride,
                         padding='VALID', use_bias=False,
                         kernel_regularizer=kernel_regularizer)

    x = normalization_fn(x, training=is_training)

    end_points['after_root'] = x

    return end_points['pre_logits'], end_points


cpc_encoder = functools.partial(encoder)
