"""Produces ratations for input images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import patch_utils
from . import patch3d_utils


def model_fn(
        data,
        mode,
        batch_size,
        crop_patches3d=None,
        perm_subset_size=8,
        serving_input_shape="None,None,None,3",
        net_params={},
):
    """Produces a loss for the jigsaw task.

    Args:
      data: Dict of inputs ("image" being the image)
      mode: model's mode: training, eval or prediction

    Returns:
      EstimatorSpec
    """
    # TODO: refactor usages
    images = data["image"]

    if crop_patches3d:
        perms, num_classes = patch3d_utils.load_permutations()
    else:
        perms, num_classes = patch_utils.load_permutations()

    labels = list(range(num_classes))

    # Selects a subset of permutation for training. There're two methods:
    #   1. For each image, selects 16 permutations independently.
    #   2. For each batch of images, selects the same 16 permutations.
    # Here we used method 2, for simplicity.
    if mode in [tf.estimator.ModeKeys.TRAIN]:
        indexs = list(range(num_classes))
        indexs = tf.random_shuffle(indexs)
        labels = indexs[:perm_subset_size]
        perms = tf.gather(perms, labels, axis=0)
        tf.logging.info("subsample %s" % perms)

    labels = tf.tile(labels, tf.shape(images)[:1])
    architecture = net_params.get("architecture", None)
    if crop_patches3d:
        return patch3d_utils.create_estimator_model(
            images,
            labels,
            perms,
            num_classes,
            mode,
            batch_size,
            architecture=architecture,
            task="jigsaw",
            serving_input_shape=serving_input_shape,
            net_params=net_params,
        )
    else:
        return patch_utils.create_estimator_model(
            images,
            labels,
            perms,
            num_classes,
            mode,
            batch_size,
            architecture=architecture,
            task="jigsaw",
            serving_input_shape=serving_input_shape,
            net_params=net_params,
        )
