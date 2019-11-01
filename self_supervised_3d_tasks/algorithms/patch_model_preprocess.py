# pylint: disable=missing-docstring
"""Preprocessing methods for self_supervised supervised representation learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from self_supervised_3d_tasks import utils as utils


def crop(image, is_training, crop_size):
    h, w, c = crop_size[0], crop_size[1], image.shape[-1]

    if is_training:
        return tf.random_crop(image, [h, w, c])
    else:
        # Central crop for now. (See Table 5 in Appendix of
        # https://arxiv.org/pdf/1703.07737.pdf for why)
        dy = (tf.shape(image)[0] - h) // 2
        dx = (tf.shape(image)[1] - w) // 2
        return tf.image.crop_to_bounding_box(image, dy, dx, h, w)


def crop3d(scan, is_training, crop_size):
    h, w, d, c = crop_size[0], crop_size[1], crop_size[2], scan.shape[-1]

    if is_training:
        return tf.random_crop(scan, [h, w, d, c])
    else:
        # Central crop for now. (See Table 5 in Appendix of
        # https://arxiv.org/pdf/1703.07737.pdf for why)
        dy = (tf.shape(scan)[0] - h) // 2
        dx = (tf.shape(scan)[1] - w) // 2
        dz = (tf.shape(scan)[2] - d) // 2

        p = tf.expand_dims(scan, 0)
        p = tf.slice(p,
                     tf.stack([0, dy, dx, dz, 0]),
                     tf.stack([-1, h, w, d, -1]))
        return tf.squeeze(p, axis=[0])


def image_to_patches(image, is_training, split_per_side, patch_jitter=0):
    """Crops split_per_side x split_per_side patches from input image.

    Args:
      image: input image tensor with shape [h, w, c].
      is_training: is training flag.
      split_per_side: split of patches per image side.
      patch_jitter: jitter of each patch from each grid.

    Returns:
      Patches tensor with shape [patch_count, hc, wc, c].
    """
    h, w, _ = image.get_shape().as_list()

    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter

    tf.logging.info(
        "Crop patches - image size: (%d, %d), split_per_side: %d, "
        "grid_size: (%d, %d), patch_size: (%d, %d), split_jitter: %d",
        h, w, split_per_side, h_grid, w_grid, h_patch, w_patch, patch_jitter)

    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):

            p = tf.image.crop_to_bounding_box(image, i * h_grid, j * w_grid, h_grid,
                                              w_grid)
            # Trick: crop a small tile from pixel cell, to avoid edge continuity.
            if h_patch < h_grid or w_patch < w_grid:
                p = crop(p, is_training, [h_patch, w_patch])

            patches.append(p)

    return tf.stack(patches)


def scan_to_patches(scan, is_training, split_per_side, patch_jitter=0):
    """Crops split_per_side x split_per_side x split_per_side patches from input scan (3d image).

    Args:
      scan: input image tensor with shape [h, w, d, c].
      is_training: is training flag.
      split_per_side: split of patches per image side.
      patch_jitter: jitter of each patch from each grid.

    Returns:
      Patches tensor with shape [patch_count, hc, wc, dc, c].
    """
    h, w, d, c = scan.get_shape().as_list()

    h_grid = h // split_per_side
    w_grid = w // split_per_side
    d_grid = d // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    d_patch = d_grid - patch_jitter

    tf.logging.info(
        "Crop patches - scan size: (%d, %d, %d), split_per_side: %d, "
        "grid_size: (%d, %d, %d), patch_size: (%d, %d, %d), split_jitter: %d",
        h, w, d, split_per_side, h_grid, w_grid, d_grid, h_patch, w_patch, d_patch, patch_jitter)

    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            for k in range(split_per_side):
                p = tf.expand_dims(scan, 0)
                p = tf.slice(p,
                             tf.stack([0, i * h_grid, j * w_grid, k * d_grid, 0]),
                             tf.stack([-1, h_grid, w_grid, d_grid, -1]))
                p = tf.squeeze(p, axis=[0])

                if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                    p = crop3d(p, is_training, [h_patch, w_patch, d_patch])

                patches.append(p)

    return tf.stack(patches)


def get_crop_patches_fn(is_training, split_per_side, patch_jitter=0):
    """Gets a function which crops split_per_side x split_per_side patches.

    Args:
      is_training: is training flag.
      split_per_side: split of patches per image side.
      patch_jitter: jitter of each patch from each grid. E.g. 255x255 input
        image with split_per_side=3 will be split into 3 85x85 grids, and
        patches are cropped from each grid with size (grid_size-patch_jitter,
        grid_size-patch_jitter).

    Returns:
      A function returns name to tensor dict. This function crops split_per_side x
      split_per_side patches from "image" tensor in input data dict.
    """

    def _crop_patches_pp(data):
        image = data["image"]

        image_to_patches_fn = functools.partial(
            image_to_patches,
            is_training=is_training,
            split_per_side=split_per_side,
            patch_jitter=patch_jitter)
        image = utils.tf_apply_to_image_or_images(image_to_patches_fn, image)

        data["image"] = image
        return data

    return _crop_patches_pp


def get_crop_patches3d_fn(is_training, split_per_side, patch_jitter=0):
    """Gets a function which crops split_per_side x split_per_side x split_per_side patches.

    Args:
      is_training: is training flag.
      split_per_side: split of patches per image side.
      patch_jitter: jitter of each patch from each grid. E.g. 255x255x255 input
        image with split_per_side=3 will be split into 3 85x85x85 grids, and
        patches are cropped from each grid with size (grid_size-patch_jitter,
        grid_size-patch_jitter, grid_size-patch_jitter).

    Returns:
      A function returns name to tensor dict. This function crops split_per_side x
      split_per_side x split_per_side patches from "scan" tensor in input data dict.
    """

    def _crop_patches_pp(data):
        scan = data["image"]

        scan_to_patches_fn = functools.partial(scan_to_patches,
                                               is_training=is_training,
                                               split_per_side=split_per_side,
                                               patch_jitter=patch_jitter)

        scan = utils.tf_apply_to_scan_or_scans(scan_to_patches_fn, scan)

        data["image"] = scan
        return data

    return _crop_patches_pp
