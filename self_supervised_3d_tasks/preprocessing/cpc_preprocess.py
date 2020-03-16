from math import sqrt

import albumentations as ab
import numpy as np
from self_supervised_3d_tasks.preprocessing.crop import crop, crop_patches
from self_supervised_3d_tasks.preprocessing.pad import pad_to_final_size_2d


def resize(batch, new_size):
    return np.array([ab.Resize(new_size, new_size)(image=image)["image"] for image in batch])


def preprocess_image(image, patch_jitter, patches_per_side, crop_size, is_training=True):
    result = []
    w, h, _ = image.shape

    if is_training:
        image = crop(image, is_training, (crop_size, crop_size))
        image = pad_to_final_size_2d(image, w)

    for patch in crop_patches(image, is_training, patches_per_side, patch_jitter):
        if is_training:
            normal_patch_size = patch.shape[0]
            patch_crop_size = int(normal_patch_size * (11.0 / 12.0))

            # patch = ab.Flip()(image=patch)["image"]
            patch = crop(patch, is_training, (patch_crop_size, patch_crop_size))
            # patch = ab.ChannelDropout(p=1.0)(image=patch)["image"]
            # patch = ab.ChannelDropout(p=1.0)(image=patch)["image"]
            # patch = ab.ToGray(p=1.0)(image=patch)["image"]  # make use of all 3 channels again for training
            patch = pad_to_final_size_2d(patch, normal_patch_size)

        else:
            # patch = crop(patch, is_training, (patch_crop_size, patch_crop_size))  # center crop here
            # patch = ab.ToGray(p=1.0)(image=patch)["image"]
            # patch = ab.PadIfNeeded(patch_crop_size + 2 * padding, patch_crop_size + 2 * padding)(image=patch)["image"]
            pass  # lets give it the most information we can get

        result.append(patch)

    return np.asarray(result)


def preprocess(batch, crop_size, patches_per_side, is_training=True):
    _, w, h, _ = batch.shape
    assert w == h, "accepting only squared images"

    patch_jitter = int(- w / (patches_per_side + 1))  # overlap half of the patch size
    return np.array([preprocess_image(image=image, patch_jitter=patch_jitter,
                                      patches_per_side=patches_per_side, crop_size=crop_size,
                                      is_training=is_training) for image in batch])


def preprocess_grid(image):
    patches_enc = []
    patches_pred = []
    labels = []

    shape = image.shape
    patch_size = int(sqrt(shape[1]))
    batch_size = shape[0]

    def get_patch_at(batch, x, y, mirror=False, predict_zero_instead_mirror=True):
        if batch < 0 or batch >= batch_size:
            return None

        if x < 0:
            if mirror:
                if predict_zero_instead_mirror:
                    return np.zeros(image[0, 0].shape)

                x = -x
            else:
                return None

        if y < 0:
            if mirror:
                if predict_zero_instead_mirror:
                    return np.zeros(image[0, 0].shape)

                y = -y
            else:
                return None

        if x >= patch_size:
            if mirror:
                if predict_zero_instead_mirror:
                    return np.zeros(image[0, 0].shape)

                x = 2 * (patch_size - 1) - x
            else:
                return None

        if y >= patch_size:
            if mirror:
                if predict_zero_instead_mirror:
                    return np.zeros(image[0, 0].shape)

                y = 2 * (patch_size - 1) - y
            else:
                return None

        return image[batch, x * patch_size + y]

    def get_patches_in_row(batch, x, x_start, y_start):
        y_min = y_start - (x_start - x)
        y_max = y_start + (x_start - x)

        patches = []
        for y in range(y_min, y_max + 1):
            patches.append(get_patch_at(batch, x, y, mirror=True))

        if x > 0:
            patches = get_patches_in_row(batch, x - 1, x_start, y_start) + patches

        return patches

    def get_patches_for(batch, x, y):
        me = get_patch_at(batch, x, y)
        others = get_patches_in_row(batch, x - 1, x, y)
        return others + [me]

    def get_following_patches(batch, x, y):
        me = get_patch_at(batch, x, y)
        if me is None:
            return []

        others = [me] + get_following_patches(batch, x + 1, y)
        return others

    end_patch_index = int(patch_size / 2) - 1  # this is the last index of the terms
    for batch_index in range(batch_size):
        for col_index in range(patch_size):
            # positive example
            terms = get_patches_for(batch_index, end_patch_index, col_index)
            predict_terms = get_following_patches(batch_index, end_patch_index + 2, col_index)
            patches_enc.append(np.stack(terms))
            patches_pred.append(np.stack(predict_terms))
            labels.append(1)

            # negative example
            r_batch = batch_index
            r_col = col_index

            while r_batch == batch_index and r_col == col_index:
                r_batch = np.random.randint(batch_size)
                r_col = np.random.randint(patch_size)

            predict_terms = get_following_patches(r_batch, end_patch_index + 2, r_col)
            patches_enc.append(np.stack(terms))
            patches_pred.append(np.stack(predict_terms))
            labels.append(0)

    return [np.stack(patches_enc), np.stack(patches_pred)], np.array(labels)
