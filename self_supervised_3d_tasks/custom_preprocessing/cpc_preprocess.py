from math import sqrt

import albumentations as ab
import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.crop import crop, crop_patches


def preprocess(batch, crop_size, split_per_side, is_training=True):
    patch_jitter = int(- crop_size / (split_per_side + 1))
    patch_crop_size = int((crop_size - patch_jitter * (split_per_side - 1)) / split_per_side * 7 / 8)
    padding = int((-2 * patch_jitter - patch_crop_size) / 2)

    return np.array([preprocess_image(image, patch_jitter, patch_crop_size, split_per_side, padding, crop_size,
                                      is_training) for image in batch])


def preprocess_image(image, patch_jitter, patch_crop_size, split_per_side, padding, crop_size, is_training=True):
    result = []
    image = crop(image, is_training, (crop_size, crop_size))

    for image in crop_patches(image, is_training, split_per_side, patch_jitter):
        image = ab.Flip()(image=image)["image"]
        image = crop(image, is_training, (patch_crop_size, patch_crop_size))
        image = ab.ChannelDropout(p=1.0)(image=image)["image"]
        image = ab.ChannelDropout(p=1.0)(image=image)["image"]
        image = ab.PadIfNeeded(patch_crop_size+2*padding, patch_crop_size+2*padding)(image=image)["image"]

        result.append(image)

    return np.asarray(result)


def preprocess_grid(image):
    patches_enc = []
    patches_pred = []
    labels = []

    shape = image.shape
    patch_size = int(sqrt(shape[1]))
    batch_size = shape[0]

    for batch_index in range(batch_size):
        for row_index in range(patch_size):
            end_patch_index = int(row_index * patch_size + int(patch_size / 2))
            patch_enc = image[batch_index, row_index * patch_size: end_patch_index, :, :]
            patches_enc.append(patch_enc)
            patches_pred.append(image[batch_index, end_patch_index + 1: (row_index + 1) * patch_size])
            labels.append(1)

            patches_enc.append(patch_enc)
            batch_index_alt = (batch_index + 1) % batch_size
            patches_pred.append(image[batch_index_alt, end_patch_index + 1: (row_index + 1) * patch_size])
            labels.append(0)

    # data["patches_enc"] = patches_enc
    # data["patches_pred"] = patches_pred
    # data["labels"] = labels

    return [np.array(patches_enc), np.array(patches_pred)], np.array(labels)
