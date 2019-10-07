"""Implements Resnet model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


################################################################################
# Basic operations building the network
################################################################################
def deconv3d(inputs, filters, kernel_size, strides, kernel_regularizer=None, use_bias=True):
    """Performs 3D deconvolution without bias and activation function."""

    return tf.layers.conv3d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=tf.truncated_normal_initializer())


def conv3d(inputs, filters, kernel_size, strides, kernel_regularizer=None, use_bias=True):
    """Performs 3D convolution without bias and activation function."""

    return tf.layers.conv3d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=tf.truncated_normal_initializer())


def batch_normalization_relu(inputs, training, batch_norm=True, instance_norm=False):
    """Performs a batch normalization followed by a ReLU6."""

    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    if batch_norm:
        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.997,
            epsilon=1e-5,
            center=True,
            scale=True,
            training=training,
            fused=True)
    elif instance_norm:
        inputs = tf.contrib.layers.instance_norm(inputs,
                                                 center=True,
                                                 scale=True,
                                                 epsilon=1e-05)

    return tf.nn.leaky_relu(inputs)


def encoding_block_layer(x, filters, blocks, strides, training, name, kernel_regularizer=None):
    """Creates one layer of encoding blocks for the model.
    Args:
        x: A tensor of size [batch, depth_in, height_in, width_in, channels].
        filters: The number of filters for the first convolution of the layer.
        block_fn: The block to use within the model.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
        name: A string name for the tensor output of the block layer.
    Returns:
        The output tensor of the block layer.
    """

    # Only the first block per block_layer uses projection_shortcut and strides
    x = conv3d(inputs=x,
               filters=filters,
               kernel_size=1,
               strides=strides,
               kernel_regularizer=kernel_regularizer)

    for _ in range(1, blocks):
        x = _residual_block(x, filters, training, 1, kernel_regularizer=kernel_regularizer)

    return tf.identity(x, name)


def decoding_block_layer(x, skip_inputs, filters, blocks, strides, training, name, kernel_regularizer=None):
    """Creates one layer of decoding blocks for the model.
    Args:
        x: A tensor of size [batch, depth_in, height_in, width_in, channels].
        skip_inputs: A tensor of size [batch, depth_in, height_in, width_in, filters].
        filters: The number of filters for the first convolution of the layer.
        block_fn: The block to use within the model.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
        name: A string name for the tensor output of the block layer.
    Returns:
        The output tensor of the block layer.
    """

    x = deconv3d(inputs=x,
                 filters=filters,
                 kernel_size=3,
                 strides=strides,
                 kernel_regularizer=kernel_regularizer)

    x = batch_normalization_relu(x, training)

    x = tf.concat([x, skip_inputs], axis=-1)

    x = conv3d(inputs=x,
               filters=filters,
               kernel_size=3,
               strides=1,
               kernel_regularizer=kernel_regularizer)

    return tf.identity(x, name)


def output_block_layer(x, training, num_classes):
    x = batch_normalization_relu(x, training)

    x = tf.layers.dropout(x, rate=0.3, training=training)

    x = conv3d(inputs=x,
               filters=num_classes,
               kernel_size=1,
               strides=1,
               use_bias=True)

    return tf.identity(x, 'output')


def _residual_block(x, filters, training, strides, kernel_regularizer=None):
    """Standard building block for residual networks with BN before convolutions.
    Args:
        x: A tensor of size [batch, depth_in, height_in, width_in, channels].
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
    Returns:
        The output tensor of the block.
    """

    shortcut = x
    x = batch_normalization_relu(x, training)

    x = conv3d(inputs=x,
               filters=filters,
               kernel_size=3,
               strides=strides,
               kernel_regularizer=kernel_regularizer)

    x = batch_normalization_relu(x, training)

    x = conv3d(inputs=x,
               filters=filters,
               kernel_size=3,
               strides=1,
               kernel_regularizer=kernel_regularizer)

    return x + shortcut


def unet_resnet(x,  # pylint: disable=missing-docstring
                is_training,
                num_encoder_layers,
                weight_decay=1e-4,
                strides=(2, 2, 2, 2, 2),
                num_classes=4,
                filters_factor=4,
                global_pool=False):
    end_points = {}

    base_filters = 8 * filters_factor  # 32: if 8 by factor, 64: if 16 by factor
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    x = conv3d(inputs=x, filters=base_filters, kernel_size=3, strides=1, kernel_regularizer=kernel_regularizer)
    x = tf.identity(x, 'initial_conv')
    initial_conv = x
    end_points['block0_filters'] = base_filters

    # *************** Encoder *******************
    skip_inputs = []
    strides = list(strides)[::-1]
    num_encoder_layers = list(num_encoder_layers)[::-1]
    for i, num_blocks in enumerate(num_encoder_layers):
        filters = base_filters * (2 ** i)
        x = encoding_block_layer(
            x=x, filters=filters, blocks=num_blocks,
            strides=strides[i], training=is_training,
            kernel_regularizer=kernel_regularizer,
            name='encode_block_layer{}'.format(i + 1))
        end_points['block' + str(i + 1)] = x
        end_points['block' + str(i + 1) + '_filters'] = filters
        skip_inputs.append(x)

    # *************** Decoder *******************
    with tf.variable_scope("decoder"):
        for i, num_blocks in reversed(list(enumerate(num_encoder_layers[1:]))):
            filters = base_filters * (2 ** i)
            x = decoding_block_layer(
                x=x, skip_inputs=skip_inputs[i],
                filters=filters, blocks=1, strides=strides[i + 1],
                training=is_training, kernel_regularizer=kernel_regularizer,
                name='decode_block_layer{}'.format(len(num_encoder_layers) - i - 1))
            end_points['block' + str(i - 1) + '_up'] = x
        x = decoding_block_layer(
            x=x, skip_inputs=initial_conv,
            filters=base_filters, blocks=1, strides=strides[-1],
            training=is_training, kernel_regularizer=kernel_regularizer,
            name='decode_block_layer{}'.format(len(num_encoder_layers) - 1))
        end_points['block' + str(len(num_encoder_layers) - 1) + '_up'] = x

        one_by_one = output_block_layer(x=x, training=is_training, num_classes=num_classes)
        end_points['one_by_one'] = one_by_one

        if global_pool:
            one_by_one = tf.reduce_mean(one_by_one, axis=[1, 2, 3], keepdims=True)
            one_by_one = tf.squeeze(one_by_one, [1, 2, 3])

    return one_by_one, end_points


unet_resnet18 = functools.partial(unet_resnet, num_encoder_layers=(1, 1, 1, 1, 1))
unet_resnet18_class = functools.partial(unet_resnet, num_encoder_layers=(1, 1, 1, 1, 1), global_pool=True)
