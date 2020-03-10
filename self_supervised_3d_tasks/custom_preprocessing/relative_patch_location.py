import albumentations as ab
from math import ceil
import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.crop import crop_patches, crop_patches_3d
from self_supervised_3d_tasks.custom_preprocessing.jigsaw_preprocess import preprocess_image_pad

# TODO: this obviously doesnt work
def preprocess_image(image, patches_per_side, patch_jitter, is_training):
    patch_size = int(image.shape[0] / patches_per_side)
    patch_count = patches_per_side**2
    center_id = int(patch_count / 2)
    # substract 1 from patch count as the id shall not point to the center
    class_id = np.random.random_integers(patch_count - 1) - 1

    if image.shape[0] != image.shape[1]:
        square_size = min(image.shape[0], image.shape[1])
        image = ab.CenterCrop(height=square_size, width=square_size)(image=image)['image']

    cropped_image = crop_patches(image, is_training, patches_per_side, patch_jitter)

    return cropped_image, class_id


def preprocess_batch(batch,  patches_per_side, patch_jitter=0, is_training=True):
    shape = batch.shape
    batch_size = shape[0]

    center_id = int(patches_per_side**2 / 2)

    patches = []
    labels = np.zeros((batch_size, patches_per_side**2 - 1))

    for batch_index in range(batch_size):
        cropped_image, class_id = preprocess_image(batch[batch_index], patches_per_side, patch_jitter, is_training)

        patch_id = class_id
        if class_id >= center_id:
            patch_id = class_id + 1

        if is_training:
            patches.append(np.array([cropped_image[center_id], cropped_image[patch_id]]))
        else:
            patches.append(cropped_image)

        labels[batch_index, class_id] = 1
    return np.array(patches), np.array(labels)

def preprocess_image_3d(image, patches_per_side, patch_jitter, is_training):
    cropped_image = crop_patches_3d(image, is_training, patches_per_side, patch_jitter)
    return np.array(cropped_image)


def preprocess_batch_3d(batch,  patches_per_side, patch_jitter=0, is_training=True):
    shape = batch.shape
    batch_size = shape[0]
    patch_count = patches_per_side ** 3

    labels = np.zeros((batch_size, patch_count - 1))
    patches = []

    center_id = int(patch_count / 2)

    for batch_index in range(batch_size):
        cropped_image = preprocess_image_3d(batch[batch_index], patches_per_side, patch_jitter, is_training)

        class_id = np.random.random_integers(patch_count - 1) - 1
        patch_id = class_id
        if class_id >= center_id:
            patch_id = class_id + 1

        if is_training:
            image_patches = np.array([cropped_image[center_id], cropped_image[patch_id]])
            patches.append(image_patches)
        else:
            # TODO: this is probably not what we want
            patches.append(np.array(cropped_image))

        labels[batch_index, class_id] = 1
    return np.array(patches), np.array(labels)
