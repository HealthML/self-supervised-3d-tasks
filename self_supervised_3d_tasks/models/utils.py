"""Helper functions for NN models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from self_supervised_3d_tasks.models import (
    vggnet,
    resnet,
    unet_resnet_2d,
    unet_resnet_3d,
)


def get_net(architecture, **kwargs):
    """ Get the specific model to be trained.
    Args:
        architecture: string selecting one of the supported architectures
        **kwargs: contains model specific arguments

    Returns:
        Network for use in #TODO: find out where I'm used

    """
    if "vgg19" == architecture:
        net = functools.partial(
            vggnet.vgg19, filters_factor=kwargs.get("filters_factor", 8)
        )
    else:
        if "resnet50" == architecture:
            net = resnet.resnet50
        elif "revnet50" == architecture:
            net = resnet.revnet50
        elif "unet_resnet50" == architecture:
            assert (
                kwargs.task == "supervised_segmentation"
            ), "unet_resnet50 is usable only with supervised_segmentation task."
            net = unet_resnet_2d.unet_resnet50
        elif "unet_resnet34" == architecture:
            assert (
                kwargs.task == "supervised_segmentation"
            ), "unet_resnet34 is usable only with supervised_segmentation task."
            net = unet_resnet_2d.unet_resnet34
        elif "unet_resnet18" == architecture:
            assert (
                kwargs.task == "supervised_segmentation"
            ), "unet_resnet18 is usable only with supervised_segmentation task."
            net = unet_resnet_2d.unet_resnet18
        elif "unet_resnet3d_class" == architecture:
            net = unet_resnet_3d.unet_resnet18_class
        elif "unet_resnet3d" == architecture:
            assert (
                kwargs.task == "supervised_segmentation"
            ), "unet_resnet3d is usable only with supervised_segmentation task."
            net = unet_resnet_3d.unet_resnet18
        else:
            raise ValueError("Unsupported architecture: %s" % architecture)

        if "unet_resnet" in architecture:
            net = functools.partial(net, filters_factor=kwargs.get("filters_factor", 4))
        else:
            net = functools.partial(
                net,
                filters_factor=kwargs.get("filters_factor", 4),
                last_relu=kwargs.get("last_relu", True),
                mode=kwargs.get("mode", "v2"),
            )

        if kwargs.task in ("jigsaw", "relative_patch_location"):
            if architecture != "unet_resnet3d_class":
                net = functools.partial(net, root_conv_stride=1, strides=(2, 2, 1))

    # Few things that are common across all models.
    net = functools.partial(
        net,
        num_classes=kwargs.get("num_classes", None),
        weight_decay=kwargs.get("weight_decay", 1e-4),
    )

    return net
