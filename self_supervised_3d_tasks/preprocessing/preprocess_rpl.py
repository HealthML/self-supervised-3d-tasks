import numpy as np
from self_supervised_3d_tasks.preprocessing.utils.crop import crop_patches, crop_patches_3d


def preprocess_image(image, patches_per_side, patch_jitter, is_training):
    cropped_image = crop_patches(image, is_training, patches_per_side, patch_jitter)
    return cropped_image


def preprocess_batch(batch,  patches_per_side, patch_jitter=0, is_training=True):
    shape = batch.shape
    batch_size = shape[0]
    patch_count = patches_per_side ** 2

    labels = np.zeros((batch_size, patch_count - 1))
    patches = []

    center_id = int(patch_count / 2)

    for batch_index in range(batch_size):
        cropped_image = preprocess_image(batch[batch_index], patches_per_side, patch_jitter, is_training)

        class_id = np.random.randint(patch_count - 1)
        patch_id = class_id
        if class_id >= center_id:
            patch_id = class_id + 1

        if is_training:
            patches.append(np.array([cropped_image[center_id], cropped_image[patch_id]]))
        else:
            patches.append(np.array(cropped_image))

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

        class_id = np.random.randint(patch_count - 1)
        patch_id = class_id
        if class_id >= center_id:
            patch_id = class_id + 1

        if is_training:
            image_patches = np.array([cropped_image[center_id], cropped_image[patch_id]])
            patches.append(image_patches)
        else:
            patches.append(np.array(cropped_image))

        labels[batch_index, class_id] = 1
    return np.array(patches), np.array(labels)
