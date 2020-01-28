import albumentations as ab
from math import ceil
import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.crop import crop, crop_patches

def preprocess_image(image, patches_per_side, patch_jitter, is_training):
    patch_count = patches_per_side * patches_per_side
    center_id = ceil(patch_count / 2)
    # substract 1 from patch count as the id shall not point to the center
    class_id = np.random.random_integers(patch_count - 1 + 1) - 1

    patch_id = class_id
    if class_id >= center_id:
        patch_id = class_id + 1

    cropped_image = crop_patches(image, is_training, patches_per_side, patch_jitter)

    # return np.array([cropped_image[center_id], cropped_image[class_id]]), class_id
    return cropped_image[class_id], class_id


def preprocess_batch(batch,  patches_per_side, patch_jitter=0, is_training=True):
    shape = batch.shape
    batch_size = shape[0]

    patches = []
    labels = np.zeros((batch_size, patches_per_side**2))

    for batch_index in range(batch_size):
        patch, class_id = preprocess_image(batch[batch_index], patches_per_side, patch_jitter, is_training)
        patches.append(patch)
        labels[batch_index, class_id] = 1
    return np.array(patches), np.array(labels)
