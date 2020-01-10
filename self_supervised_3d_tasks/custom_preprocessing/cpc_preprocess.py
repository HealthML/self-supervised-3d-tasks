import albumentations as ab
import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.crop import crop, crop_patches


def preprocess(image, crop_size, split_per_side, is_training=True):
    patch_jitter = int(- crop_size / (split_per_side + 1))
    patch_crop_size = int((crop_size - patch_jitter * (split_per_side - 1)) / split_per_side * 7 / 8)
    padding = int((-2 * patch_jitter - patch_crop_size) / 2)

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
