import random

import albumentations as ab
import numpy as np

from self_supervised_3d_tasks.custom_preprocessing.crop import crop_patches, crop_patches_3d
from self_supervised_3d_tasks.custom_preprocessing.pad import pad_to_final_size_3d


def preprocess_image(image, is_training, split_per_side, patch_jitter, permutations, mode3d):
    label = random.randint(0, len(permutations) - 1)

    if mode3d:
        patches = crop_patches_3d(image, is_training, split_per_side, patch_jitter)
    else:
        patches = crop_patches(image, is_training, split_per_side, patch_jitter)

    b = np.zeros((len(permutations),))
    b[label] = 1

    return np.array(patches)[np.array(permutations[label])], np.array(b)


def preprocess(batch, split_per_side, patch_jitter, permutations, is_training=True, mode3d=False):
    xs = []
    ys = []

    for image in batch:
        x, y = preprocess_image(image, is_training, split_per_side, patch_jitter, permutations, mode3d)
        xs.append(x)
        ys.append(y)

    xs = np.stack(xs)
    ys = np.stack(ys)

    return xs, ys


def preprocess_image_crop_only(image, split_per_side, is_training, mode3d):
    if mode3d:
        patches = crop_patches_3d(image, is_training, split_per_side, 0)
    else:
        patches = crop_patches(image, is_training, split_per_side, 0)
    return np.stack(patches)


def preprocess_crop_only(batch, split_per_side, is_training=True, mode3d=False):
    xs = []

    for image in batch:
        x = preprocess_image_crop_only(image, split_per_side, is_training, mode3d)
        xs.append(x)

    return np.stack(xs)


def preprocess_image_pad(patches, patch_dim, mode3d):
    result = []

    for patch in patches:
        if mode3d:
            patch = pad_to_final_size_3d(patch, patch_dim)
        else:
            patch = ab.PadIfNeeded(patch_dim, patch_dim)(image=patch)["image"]

        result.append(patch)

    return np.stack(result)


def preprocess_pad(batch, patch_dim, mode3d=False):
    xs = []

    for patches in batch:
        x = preprocess_image_pad(patches, patch_dim, mode3d)
        xs.append(x)

    return np.stack(xs)
