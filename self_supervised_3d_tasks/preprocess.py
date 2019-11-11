# pylint: disable=missing-docstring
"""Preprocessing methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random

import tensorflow as tf

from self_supervised_3d_tasks import inception_preprocessing, utils
from self_supervised_3d_tasks.algorithms import patch_model_preprocess as pp_lib


def get_inception_preprocess(is_training, im_size):
    def _inception_preprocess(data):
        data["image"] = inception_preprocessing.preprocess_image(
            data["image"],
            im_size[0],
            im_size[1],
            is_training,
            add_image_summaries=False,
            crop_image=False,
        )
        return data

    return _inception_preprocess


def get_resize_small(smaller_size):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio."""

    def _resize_small_pp(data):
        image = data["image"]
        # A single image: HWC
        # A batch of images: BHWC
        h, w = tf.shape(image)[-3], tf.shape(image)[-2]

        # Figure out the necessary h/w.
        ratio = tf.to_float(smaller_size) / tf.to_float(tf.minimum(h, w))
        h = tf.to_int32(tf.round(tf.to_float(h) * ratio))
        w = tf.to_int32(tf.round(tf.to_float(w) * ratio))

        # NOTE: use align_corners=False for AREA resize, but True for Bilinear.
        # See also https://github.com/tensorflow/tensorflow/issues/6720
        static_rank = len(image.get_shape().as_list())
        if static_rank == 3:  # A single image: HWC
            data["image"] = tf.image.resize_area(image[None], [h, w])[0]
        elif static_rank == 4:  # A batch of images: BHWC
            data["image"] = tf.image.resize_area(image, [h, w])
        return data

    return _resize_small_pp


def get_crop(is_training, crop_size):
    """Returns a random (or central at test-time) crop of `crop_size`."""

    def _crop_pp(data):
        crop_fn = functools.partial(
            pp_lib.crop, is_training=is_training, crop_size=crop_size
        )
        data["image"] = utils.tf_apply_to_image_or_images(crop_fn, data["image"])

        return data

    return _crop_pp


def get_inception_crop(is_training, **kw):
    # kw of interest are: aspect_ratio_range, area_range.
    # Note that image is not resized yet here.
    def _inception_crop_pp(data):
        if is_training:
            image = data["image"]
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                tf.zeros([0, 0, 4], tf.float32),
                use_image_if_no_bounding_boxes=True,
                **kw
            )
            data["image"] = tf.slice(image, begin, size)
            # Unfortunately, the above operation loses the depth-dimension. So we need
            # to Restore it the manual way.
            data["image"].set_shape([None, None, image.shape[-1]])
        return data

    return _inception_crop_pp


def get_pad(padding, mode):
    def _make_padding(data):
        data["image"] = utils.tf_apply_to_image_or_images(
            lambda img: tf.pad(img, tf.constant(padding), mode), data["image"]
        )

        return data

    return _make_padding


def get_random_flip_lr(is_training):
    def _random_flip_lr_pp(data):
        if is_training:
            data["image"] = utils.tf_apply_to_image_or_images(
                tf.image.random_flip_left_right, data["image"]
            )
        return data

    return _random_flip_lr_pp


def get_random_flip_ud(is_training):
    def _random_flip_ud_pp(data):
        if is_training:
            data["image"] = utils.tf_apply_to_image_or_images(
                tf.image.random_flip_up_down, data["image"]
            )
        return data

    return _random_flip_ud_pp


def get_resize_preprocess(im_size, randomize_resize_method=False):
    def _resize(image, method, align_corners):
        def _process():
            # The resized_images are of type float32 and might fall outside of range
            # [0, 255].
            resized = tf.cast(
                tf.image.resize_images(
                    image, im_size, method, align_corners=align_corners
                ),
                dtype=tf.float32,
            )
            return resized

        return _process

    def _resize_pp(data):
        im = data["image"]

        if randomize_resize_method:
            # pick random resizing method
            r = tf.random_uniform([], 0, 3, dtype=tf.int32)
            im = tf.case(
                {
                    tf.equal(r, tf.cast(0, r.dtype)): _resize(
                        im, tf.image.ResizeMethod.BILINEAR, True
                    ),
                    tf.equal(r, tf.cast(1, r.dtype)): _resize(
                        im, tf.image.ResizeMethod.NEAREST_NEIGHBOR, True
                    ),
                    tf.equal(r, tf.cast(2, r.dtype)): _resize(
                        im, tf.image.ResizeMethod.BICUBIC, True
                    ),
                    # NOTE: use align_corners=False for AREA resize, but True for the
                    # others. See https://github.com/tensorflow/tensorflow/issues/6720
                    tf.equal(r, tf.cast(3, r.dtype)): _resize(
                        im, tf.image.ResizeMethod.AREA, False
                    ),
                }
            )
        else:
            im = tf.image.resize_images(im, im_size)
        data["image"] = im
        return data

    return _resize_pp


def get_resize_segmentation_preprocess(im_size, randomize_resize_method=False):
    def _resize_image_mask(image, mask, method_im, method_mask, align_corners):
        def _process():
            # The resized_images are of type float32 and might fall outside of range
            # [0, 255].
            resized_im = tf.cast(
                tf.image.resize_images(
                    image, im_size, method_im, align_corners=align_corners
                ),
                dtype=tf.float32,
            )
            resized_mask = tf.cast(
                tf.image.resize_images(
                    mask, im_size, method_mask, align_corners=align_corners
                ),
                dtype=tf.float32,
            )
            return resized_im, resized_mask

        return _process

    def _resize_pp(data):
        im = data["image"]
        mask = data["label"]
        mask = tf.expand_dims(mask, axis=-1)
        if randomize_resize_method:
            # pick random resizing method
            r = tf.random_uniform([], 0, 3, dtype=tf.int32)
            im, mask = tf.case(
                {
                    tf.equal(r, tf.cast(0, r.dtype)): _resize_image_mask(
                        im,
                        mask,
                        tf.image.ResizeMethod.BILINEAR,
                        tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        True,
                    ),
                    tf.equal(r, tf.cast(1, r.dtype)): _resize_image_mask(
                        im,
                        mask,
                        tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        True,
                    ),
                    tf.equal(r, tf.cast(2, r.dtype)): _resize_image_mask(
                        im,
                        mask,
                        tf.image.ResizeMethod.BICUBIC,
                        tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        True,
                    ),
                    # NOTE: use align_corners=False for AREA resize, but True for the
                    # others. See https://github.com/tensorflow/tensorflow/issues/6720
                    tf.equal(r, tf.cast(3, r.dtype)): _resize_image_mask(
                        im,
                        mask,
                        tf.image.ResizeMethod.AREA,
                        tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        False,
                    ),
                }
            )
        else:
            im = tf.image.resize_images(im, im_size)
            mask = tf.image.resize_images(
                mask, im_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        mask = tf.squeeze(mask)
        data["image"] = im
        data["label"] = mask
        return data

    return _resize_pp


def get_rotate_preprocess():
    """Returns a function that does 90deg rotations and sets according labels."""

    def _rotate_pp(data):
        data["label"] = tf.constant([0, 1, 2, 3])
        # We use our own instead of tf.image.rot90 because that one broke
        # internally shortly before deadline...
        data["image"] = tf.stack(
            [
                data["image"],
                tf.transpose(tf.reverse_v2(data["image"], [1]), [1, 0, 2]),
                tf.reverse_v2(data["image"], [0, 1]),
                tf.reverse_v2(tf.transpose(data["image"], [1, 0, 2]), [1]),
            ]
        )
        return data

    return _rotate_pp


def get_rotate3d_preprocess():
    """Returns a function that does 90deg rotations and sets according labels."""

    def _rotate_pp(data):
        # data["label"] = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        data["label"] = tf.constant([0, 1, 2, 3, 4, 5, 6])

        data["image"] = tf.stack(
            [
                data["image"],
                tf.transpose(tf.reverse_v2(data["image"], [1]), [1, 0, 2, 3]),
                # tf.reverse_v2(data["image"], [0, 1]), # 180 degrees on z axis
                tf.reverse_v2(tf.transpose(data["image"], [1, 0, 2, 3]), [1]),
                tf.transpose(tf.reverse_v2(data["image"], [1]), [0, 2, 1, 3]),
                # tf.reverse_v2(data["image"], [1, 2]), # 180 degrees on x axis
                tf.reverse_v2(tf.transpose(data["image"], [0, 2, 1, 3]), [1]),
                tf.transpose(tf.reverse_v2(data["image"], [0]), [2, 1, 0, 3]),
                # tf.reverse_v2(data["image"], [0, 2]), # 180 degrees on y axis
                tf.reverse_v2(tf.transpose(data["image"], [2, 1, 0, 3]), [0]),
            ]
        )
        return data

    return _rotate_pp


def get_value_range_preprocess(vmin=-1, vmax=1, dtype=tf.float32):
    """Returns a function that sends [0,255] image to [vmin,vmax]."""

    def _value_range_pp(data):
        img = tf.cast(data["image"], dtype)
        img = vmin + (img / tf.constant(255.0, dtype)) * (vmax - vmin)
        data["image"] = img
        return data

    return _value_range_pp


def get_duplicate_channels3d_preprocess():
    """Returns a function that duplicates the input image channels, e.g. [128,128,2] --> [128,128,4]."""

    def _duplicate_channels_pp(data):
        img = data["image"]
        img = tf.tile(img, [1, 1, 1, 2])  # duplicating the channels
        data["image"] = img
        return data

    return _duplicate_channels_pp


def get_standardization_preprocess():
    def _standardization_pp(data):
        # Trick: normalize each patch to avoid low level statistics.
        data["image"] = utils.tf_apply_to_image_or_images(
            tf.image.per_image_standardization, data["image"]
        )
        return data

    return _standardization_pp


def get_inception_preprocess_patches(is_training, resize_size, num_of_patches):
    def _inception_preprocess_patches(data):
        patches = []
        for _ in range(num_of_patches):
            patches.append(
                inception_preprocessing.preprocess_image(
                    data["image"],
                    resize_size[0],
                    resize_size[1],
                    is_training,
                    add_image_summaries=False,
                    crop_image=False,
                )
            )
        patches = tf.stack(patches)
        data["image"] = patches
        return data

    return _inception_preprocess_patches


def get_inception_preprocess_patches3d(num_of_patches, fast_mode=True):
    def _inception_preprocess_patches(data):
        patches = []
        scan = data["image"]
        if not fast_mode:
            num_of_instances = num_of_patches - 4
        else:
            num_of_instances = num_of_patches
        for _ in range(num_of_instances):
            rand_flip_axis = random.randint(0, 3)
            if rand_flip_axis == 0:
                distorted_scan = scan
            elif rand_flip_axis == 1:
                distorted_scan = (tf.reverse_v2(scan, [0, 1]),)  # 180 degrees on z axis
            elif rand_flip_axis == 2:
                distorted_scan = (tf.reverse_v2(scan, [1, 2]),)  # 180 degrees on x axis
            else:
                distorted_scan = (tf.reverse_v2(scan, [0, 2]),)  # 180 degrees on y axis

            num_cases = 1 if fast_mode else 4
            for case in range(num_cases):

                distorted_scan = distort_color3d(distorted_scan)
                if len(distorted_scan.get_shape().as_list()) > 4:
                    distorted_scan = tf.squeeze(distorted_scan, axis=[0])
                patches.append(distorted_scan)

        patches = tf.stack(patches)
        data["image"] = patches
        return data

    return _inception_preprocess_patches


def distort_color3d(scan):
    x = scan
    # adjust_brightness
    max_delta = 32.0 / 255.0
    delta = tf.random_uniform([], -max_delta, max_delta)
    x = tf.add(x, tf.cast(delta, tf.float32))

    # adjust_contrast
    lower = 0.5
    upper = 1.5
    contrast_factor = tf.random_uniform([], lower, upper)
    x_mean = tf.reduce_mean(x, keep_dims=True)
    x = tf.add(
        tf.multiply(tf.subtract(x, tf.reduce_mean(x, keep_dims=True)), contrast_factor),
        x_mean,
    )

    return tf.clip_by_value(x, 0.0, 1.0)


def get_drop_all_channels_but_one_preprocess():
    def _drop_all_channels_but_one(idx):
        # KEEP THE CHANNEL POSITION
        # x = np.zeros((56,56,3))
        # x[:,:,idx] = 1
        # mask = tf.constant(x,dtype=tf.float32)
        # return lambda image: tf.multiply(image, mask)

        return lambda image: tf.tile(image[:, :, idx : idx + 1], [1, 1, 3])

    def _drop_all_channels_but_one_pp(data):
        data["image"] = utils.tf_apply_to_image_or_images(
            lambda img: utils.tf_apply_many_with_probability(
                [1.0 / 3.0] * 3,
                [(_drop_all_channels_but_one(a)) for a in range(3)],
                img,
            ),
            data["image"],
        )
        return data

    return _drop_all_channels_but_one_pp


def get_to_gray_preprocess(grayscale_probability):
    def _to_gray(image):
        # Transform to grayscale by taking the mean of RGB.
        return tf.tile(tf.reduce_mean(image, axis=2, keepdims=True), [1, 1, 3])

    def _to_gray_pp(data):
        data["image"] = utils.tf_apply_to_image_or_images(
            lambda img: utils.tf_apply_with_probability(  # pylint:disable=g-long-lambda
                grayscale_probability, _to_gray, img
            ),
            data["image"],
        )
        return data

    return _to_gray_pp


def get_preprocess_fn(fn_names, is_training, **dependend_params):
    """Returns preprocessing function.

    Args:
      fn_names: name of a preprocessing function.
      is_training: Whether this should be run in train or eval mode.
    Returns:
      preprocessing function

    Raises:
      ValueError: if preprocessing function name is unknown
    """

    def _fn(data):
        def expand(fn_name):
            if fn_name == "plain_preprocess":
                yield lambda x: x
            elif fn_name == "duplicate_channels3d":
                yield get_duplicate_channels3d_preprocess()
            elif fn_name == "0_to_1":
                yield get_value_range_preprocess(0, 1)
            elif fn_name == "-1_to_1":
                yield get_value_range_preprocess(-1, 1)
            elif fn_name == "resize":
                yield get_resize_preprocess(
                    utils.str2intlist(dependend_params["resize_size"], 2),
                    is_training
                    and dependend_params.get("randomize_resize_method", False),
                )
            elif fn_name == "resize_segmentation":
                yield get_resize_segmentation_preprocess(
                    utils.str2intlist(dependend_params["resize_size"], 2),
                    is_training
                    and dependend_params.get("randomize_resize_method", False),
                )
            elif fn_name == "resize_small":
                yield get_resize_small(dependend_params["smaller_size"])
            elif fn_name == "crop":
                yield get_crop(
                    is_training, utils.str2intlist(dependend_params["crop_size"], 2)
                )
            elif fn_name == "central_crop":
                yield get_crop(
                    False, utils.str2intlist(dependend_params["crop_size"], 2)
                )
            elif fn_name == "inception_crop":
                yield get_inception_crop(is_training)
            elif fn_name == "pad":
                yield get_pad(dependend_params["padding"], dependend_params["padding_mode"])
            elif fn_name == "flip_lr":
                yield get_random_flip_lr(is_training)
            elif fn_name == "flip_ud":
                yield get_random_flip_ud(is_training)
            elif fn_name == "crop_inception_preprocess_patches":
                yield get_inception_preprocess_patches(
                    is_training,
                    utils.str2intlist(dependend_params["resize_size"], 2),
                    dependend_params["num_of_inception_patches"],
                )
            elif fn_name == "crop_inception_preprocess_patches3d":
                yield get_inception_preprocess_patches3d(
                    dependend_params["num_of_inception_patches"],
                    fast_mode=dependend_params["fast_mode"],
                )
            elif fn_name == "to_gray":
                yield get_to_gray_preprocess(
                    dependend_params.get("grayscale_probability", 1.0)
                )
            elif fn_name == "drop_all_channels_but_one":
                yield get_drop_all_channels_but_one_preprocess()
            elif fn_name == "crop_patches":
                yield pp_lib.get_crop_patches_fn(
                    is_training,
                    split_per_side=dependend_params["splits_per_side"],
                    patch_jitter=dependend_params.get("patch_jitter", 0),
                )
            elif fn_name == "crop_patches3d":
                yield pp_lib.get_crop_patches3d_fn(
                    is_training,
                    split_per_side=dependend_params["splits_per_side"],
                    patch_jitter=dependend_params.get("patch_jitter", 0),
                )
            elif fn_name == "standardization":
                yield get_standardization_preprocess()
            elif fn_name == "rotate":
                yield get_rotate_preprocess()
            elif fn_name == "rotate3d":
                yield get_rotate3d_preprocess()

            # Below this line specific combos decomposed.
            # It would be nice to move them to the configs at some point.

            elif fn_name == "inception_preprocess":
                yield get_inception_preprocess(
                    is_training, utils.str2intlist(dependend_params["resize_size"], 2)
                )
            else:
                raise ValueError("Not supported preprocessing %s" % fn_name)

        # Apply all the individual steps in sequence.
        tf.logging.info("Data before pre-processing:\n%s", data)
        for fn_name in fn_names:
            print(">>>>>", fn_name)
            for p in expand(fn_name.strip()):
                data = p(data, dependend_params)
                tf.logging.info("Data after `%s`:\n%s", p, data)
        return data

    return _fn
