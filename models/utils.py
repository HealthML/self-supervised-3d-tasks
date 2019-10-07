"""Helper functions for NN models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import absl.flags as flags

import models
from models import unet_resnet_2d, unet_resnet_3d, resnet, vggnet

FLAGS = flags.FLAGS


def get_net(num_classes=None):  # pylint: disable=missing-docstring
    architecture = FLAGS.architecture

    if 'vgg19' == architecture:
        net = functools.partial(
            models.vggnet.vgg19,
            filters_factor=FLAGS.get_flag_value('filters_factor', 8))
    else:
        if 'resnet50' == architecture:
            net = models.resnet.resnet50
        elif 'revnet50' == architecture:
            net = models.resnet.revnet50
        elif 'unet_resnet50' == architecture:
            assert FLAGS.task == 'supervised_segmentation', 'unet_resnet50 is usable only with supervised_segmentation task.'
            net = models.unet_resnet_2d.unet_resnet50
        elif 'unet_resnet34' == architecture:
            assert FLAGS.task == 'supervised_segmentation', 'unet_resnet34 is usable only with supervised_segmentation task.'
            net = models.unet_resnet_2d.unet_resnet34
        elif 'unet_resnet18' == architecture:
            assert FLAGS.task == 'supervised_segmentation', 'unet_resnet18 is usable only with supervised_segmentation task.'
            net = models.unet_resnet_2d.unet_resnet18
        elif 'unet_resnet3d_class' == architecture:
            net = models.unet_resnet_3d.unet_resnet18_class
        elif 'unet_resnet3d' == architecture:
            assert FLAGS.task == 'supervised_segmentation', 'unet_resnet3d is usable only with supervised_segmentation task.'
            net = models.unet_resnet_3d.unet_resnet18
        else:
            raise ValueError('Unsupported architecture: %s' % architecture)

        if 'unet_resnet' in architecture:
            net = functools.partial(
                net,
                filters_factor=FLAGS.get_flag_value('filters_factor', 4))
        else:
            net = functools.partial(
                net,
                filters_factor=FLAGS.get_flag_value('filters_factor', 4),
                last_relu=FLAGS.get_flag_value('last_relu', True),
                mode=FLAGS.get_flag_value('mode', 'v2'))

        if FLAGS.task in ('jigsaw', 'relative_patch_location'):
            if FLAGS.architecture != 'unet_resnet3d_class':
                net = functools.partial(net, root_conv_stride=1, strides=(2, 2, 1))

    # Few things that are common across all models.
    net = functools.partial(
        net, num_classes=num_classes,
        weight_decay=FLAGS.get_flag_value('weight_decay', 1e-4))

    return net
