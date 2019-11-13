"""Produces ratations for input images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from self_supervised_3d_tasks.algorithms import patch_utils, patch3d_utils


# FLAGS = tf.flags.FLAGS


def model_fn(
        data, mode, batch_size, crop_patches3d=False, serving_input_shape="None,None,None,3"
):
    """Produces a loss for the relative patch location task.

    Args:
      data: Dict of inputs ("image" being the image)
      mode: model's mode: training, eval or prediction

    Returns:
      EstimatorSpec
    """
    # TODO: refactor usages
    images = data["image"]

    # Patch locations
    if crop_patches3d:
        perms, num_classes = patch3d_utils.generate_patch_locations()
    else:
        perms, num_classes = patch_utils.generate_patch_locations()

    labels = tf.tile(list(range(num_classes)), tf.shape(images)[:1])

    if crop_patches3d:
        return patch3d_utils.create_estimator_model(
            images,
            labels,
            perms,
            num_classes,
            mode,
            batch_size,
            serving_input_shape=serving_input_shape,
        )
    else:
        return patch_utils.create_estimator_model(
            images,
            labels,
            perms,
            num_classes,
            mode,
            batch_size,
            serving_input_shape=serving_input_shape,
        )
