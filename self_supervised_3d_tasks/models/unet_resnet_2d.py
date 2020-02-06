"""Implements Resnet model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


def get_shape_as_list(x):
    return x.get_shape().as_list()


def fixed_padding(x, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return x


def batch_norm(x, training):
    return tf.layers.batch_normalization(x, fused=True, training=training)


def identity_norm(x, training):
    del training
    return x


def bottleneck_v1_block(
        x,
        filters,
        training,  # pylint: disable=missing-docstring
        strides=1,
        activation_fn=tf.nn.relu,
        normalization_fn=batch_norm,
        kernel_regularizer=None,
):
    # Record input tensor, such that it can be used later in as skip-connection
    x_shortcut = x

    # Project input if necessary
    if (strides > 1) or (filters != x.shape[-1]):
        x_shortcut = tf.layers.conv2d(
            x_shortcut,
            filters=filters,
            kernel_size=1,
            strides=strides,
            kernel_regularizer=kernel_regularizer,
            padding="SAME",
        )
        x_shortcut = normalization_fn(x_shortcut, training=training)

    # First convolution
    # Note, that unlike original Resnet paper we never use stride in the first
    # convolution. Instead, we apply stride in the second convolution. The reason
    # is that the first convolution has kernel of size 1x1, which results in
    # information loss when combined with stride bigger than one.
    x = tf.layers.conv2d(
        x,
        filters=filters // 4,
        kernel_size=1,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )
    x = normalization_fn(x, training=training)
    x = activation_fn(x)

    # Second convolution
    x = fixed_padding(x, kernel_size=3)
    x = tf.layers.conv2d(
        x,
        filters=filters // 4,
        strides=strides,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
        padding="VALID",
    )
    x = normalization_fn(x, training=training)
    x = activation_fn(x)

    # Third convolution
    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=1,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )
    x = normalization_fn(x, training=training)

    # Skip connection
    x = x_shortcut + x
    x = activation_fn(x)

    return x


def residual_block(
        x,
        filters,
        training,  # pylint: disable=missing-docstring
        strides=1,
        activation_fn=tf.nn.relu,
        normalization_fn=batch_norm,
        kernel_regularizer=None,
):
    # Record input tensor, such that it can be used later in as skip-connection
    x_shortcut = x

    # Project input if necessary
    if (strides > 1) or (filters != x.shape[-1]):
        x_shortcut = tf.layers.conv2d(
            x_shortcut,
            filters=filters,
            kernel_size=1,
            strides=strides,
            kernel_regularizer=kernel_regularizer,
            padding="SAME",
        )
        x_shortcut = normalization_fn(x_shortcut, training=training)

    # First convolution
    # Note, that unlike original Resnet paper we never use stride in the first
    # convolution. Instead, we apply stride in the second convolution. The reason
    # is that the first convolution has kernel of size 1x1, which results in
    # information loss when combined with stride bigger than one.
    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )
    x = normalization_fn(x, training=training)
    x = activation_fn(x)

    # Second convolution
    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=3,
        strides=strides,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )
    x = normalization_fn(x, training=training)

    # Skip connection
    x = x_shortcut + x
    x = activation_fn(x)

    return x


def decoder_upsampling(
        x,
        skip,
        filters,
        training,
        use_batchnorm=False,
        normalization_fn=batch_norm,
        activation_fn=tf.nn.relu,
        kernel_regularizer=None,
):
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    if skip is not None:
        x = tf.concat([x, skip], axis=-1)
    if use_batchnorm:
        x = normalization_fn(x, training=training)

    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )

    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )
    x = activation_fn(x)

    return x


def decoder_transpose(
        x,
        skip,
        filters,
        training,
        use_batchnorm=True,
        normalization_fn=batch_norm,
        activation_fn=tf.nn.relu,
        kernel_regularizer=None,
):
    x = tf.layers.conv2d_transpose(
        x, filters=filters, kernel_size=3, strides=(2, 2), padding="SAME"
    )

    if use_batchnorm:
        x = normalization_fn(x, training=training)

    x = activation_fn(x)

    if skip is not None:
        x = tf.concat([x, skip], axis=-1)

    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=3,
        kernel_regularizer=kernel_regularizer,
        padding="SAME",
    )

    return x


def unet_resnet(
        x,  # pylint: disable=missing-docstring
        is_training,
        num_encoder_layers,
        block,
        strides=(2, 2, 2, 2),
        num_classes=4,
        filters_factor=4,
        weight_decay=1e-4,
        include_root_block=False,
        root_conv_size=7,
        root_conv_stride=2,
        root_pool_size=3,
        root_pool_stride=2,
        activation_fn=tf.nn.relu,
        normalization_fn=batch_norm,
):
    end_points = {}

    decoder_block = decoder_transpose  # decoder_upsampling  # can be decoder_transpose
    filters = 16 * filters_factor
    end_points["block0_filters"] = filters

    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    if include_root_block:
        x = fixed_padding(x, kernel_size=root_conv_size)
        x = tf.layers.conv2d(
            x,
            filters=filters,
            kernel_size=root_conv_size,
            strides=root_conv_stride,
            padding="VALID",
            kernel_regularizer=kernel_regularizer,
        )

        x = normalization_fn(x, training=is_training)
        x = activation_fn(x)

        x = fixed_padding(x, kernel_size=root_pool_size)
        x = tf.layers.max_pooling2d(
            x, pool_size=root_pool_size, strides=root_pool_stride, padding="VALID"
        )
        end_points["after_root"] = x

    params = {
        "activation_fn": activation_fn,
        "normalization_fn": normalization_fn,
        "training": is_training,
        "kernel_regularizer": kernel_regularizer,
    }

    # *************** Encoder *******************
    strides = list(strides)[::-1]
    num_encoder_layers = list(num_encoder_layers)[::-1]

    if block == bottleneck_v1_block:
        filters *= 4
    for _ in range(num_encoder_layers.pop()):
        x = block(x, filters, strides=1, **params)
    end_points["block1"] = x
    end_points["block1_filters"] = filters

    filters *= 2
    x = block(x, filters, strides=strides.pop(), **params)
    for _ in range(num_encoder_layers.pop() - 1):
        x = block(x, filters, strides=1, **params)
    end_points["block2"] = x
    end_points["block2_filters"] = filters

    filters *= 2
    x = block(x, filters, strides=strides.pop(), **params)
    for _ in range(num_encoder_layers.pop() - 1):
        x = block(x, filters, strides=1, **params)
    end_points["block3"] = x
    end_points["block3_filters"] = filters

    filters *= 2
    x = block(x, filters, strides=strides.pop(), **params)
    for _ in range(num_encoder_layers.pop() - 1):
        x = block(x, filters, strides=1, **params)
    end_points["block4"] = x
    end_points["block4_filters"] = filters

    filters *= 2
    x = block(x, filters, strides=strides.pop(), **params)
    for _ in range(num_encoder_layers.pop() - 1):
        x = block(x, filters, strides=1, **params)
    end_points["block5"] = x
    end_points["block5_filters"] = filters

    # *************** Decoder *******************
    with tf.variable_scope("decoder"):
        x = decoder_block(
            x,
            skip=end_points["block4"],
            filters=end_points["block5_filters"],
            training=is_training,
        )
        end_points["block5_up"] = x

        x = decoder_block(
            x,
            skip=end_points["block3"],
            filters=end_points["block4_filters"],
            training=is_training,
        )
        end_points["block4_up"] = x

        x = decoder_block(
            x,
            skip=end_points["block2"],
            filters=end_points["block3_filters"],
            training=is_training,
        )
        end_points["block3_up"] = x

        x = decoder_block(
            x,
            skip=end_points["block1"],
            filters=end_points["block2_filters"],
            training=is_training,
        )
        end_points["block2_up"] = x

        # x = decoder_block(x, skip=None, filters=end_points['block1_filters'], training=is_training)
        # end_points['block1_up'] = x
        #
        # x = decoder_block(x, skip=None, filters=end_points['block0_filters'], training=is_training)
        # end_points['block0_up'] = x

        one_by_one = tf.layers.conv2d(
            x, filters=num_classes, kernel_size=1, padding="SAME", name="fc"
        )
        end_points["one_by_one"] = one_by_one

    return one_by_one, end_points


unet_resnet50 = functools.partial(
    unet_resnet, num_encoder_layers=(3, 4, 6, 3, 3), block=bottleneck_v1_block
)
unet_resnet34 = functools.partial(
    unet_resnet, num_encoder_layers=(3, 4, 6, 3, 3), block=residual_block
)
unet_resnet18 = functools.partial(
    unet_resnet, num_encoder_layers=(2, 2, 2, 2, 2), block=residual_block
)
