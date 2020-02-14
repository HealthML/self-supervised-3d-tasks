from math import sqrt

import albumentations as ab
import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.crop import crop, crop_patches, crop_patches_3d, crop_3d


def pad_to_final_size(volume, w):
    dim = volume.shape[0]
    f1 = int((w - dim) / 2)
    f2 = w - f1

    return np.pad(volume, (f1, f2), "edge")


def preprocess_volume_3d(volume, crop_size, split_per_side, patch_overlap, is_training=True):
    result = []
    w, _, _, _ = volume.shape

    if is_training:
        volume = crop_3d(volume, is_training, (crop_size, crop_size, crop_size))
        volume = pad_to_final_size(volume, w)

    for patch in crop_patches_3d(volume, is_training, split_per_side, -patch_overlap):
        if is_training:
            normal_patch_size = patch.shape[0]
            patch_crop_size = int(normal_patch_size * (7.0 / 8.0))

            do_flip = np.random.choice([False, True])
            if do_flip:
                patch = np.flip(patch, 0)

            patch = crop_3d(patch, is_training, (patch_crop_size, patch_crop_size, patch_crop_size))
            patch = pad_to_final_size(patch, normal_patch_size)

        else:
            # patch = crop(patch, is_training, (patch_crop_size, patch_crop_size))  # center crop here
            # patch = ab.ToGray(p=1.0)(image=patch)["image"]
            # patch = ab.PadIfNeeded(patch_crop_size + 2 * padding, patch_crop_size + 2 * padding)(image=patch)["image"]
            pass  # lets give it the most information we can get

        result.append(patch)

    return np.asarray(result)


def preprocess_3d(batch, crop_size, split_per_side, patch_overlap, is_training=True):
    _, w, h, d, _ = batch.shape
    assert w == h and h == d, "accepting only cube volumes"

    return np.stack([preprocess_volume_3d(volume, crop_size, split_per_side, patch_overlap, is_training=True)
                     for volume in batch])


def preprocess_grid_3d(image):
    patches_enc = []
    patches_pred = []
    labels = []

    shape = image.shape
    batch_size = shape[0]
    n_patches_one_dim = int(np.cbrt(shape[1]))

    def get_patch_at(batch, x, y, z, mirror=False):
        if batch < 0 or batch >= batch_size:
            return None

        if x < 0:
            if mirror:
                x = -x
            else:
                return None

        if y < 0:
            if mirror:
                y = -y
            else:
                return None

        if z < 0:
            if mirror:
                z = -z
            else:
                return None

        if x >= n_patches_one_dim:
            if mirror:
                x = 2 * (n_patches_one_dim - 1) - x
            else:
                return None

        if y >= n_patches_one_dim:
            if mirror:
                y = 2 * (n_patches_one_dim - 1) - y
            else:
                return None

        if z >= n_patches_one_dim:
            if mirror:
                z = 2 * (n_patches_one_dim - 1) - z
            else:
                return None

        return image[batch, x * n_patches_one_dim * n_patches_one_dim + y * n_patches_one_dim + z]

    # this function will collect all the patches at one x-level that should be used in the terms
    # in comparison to the x_start, we know how far we have to expand, forming a pyramid in 3D
    def get_patches_in_row(batch, x, x_start, y_start, z_start):
        y_min = y_start - (x_start - x)
        y_max = y_start + (x_start - x)

        z_min = z_start - (x_start - x)
        z_max = z_start + (x_start - x)

        patches = []
        for y in range(y_min, y_max + 1):
            for z in range(z_min, z_max + 1):
                patches.append(get_patch_at(batch, x, y, z, mirror=True))

        if x > 0:
            patches = get_patches_in_row(batch, x - 1, x_start, y_start, z_start) + patches

        return patches

    def get_patches_for(batch, x, y, z):
        me = get_patch_at(batch, x, y, z)
        others = get_patches_in_row(batch, x - 1, x, y, z)
        return others + [me]

    def get_following_patches(batch, x, y, z):
        me = get_patch_at(batch, x, y, z)
        if me is None:
            return []

        others = [me] + get_following_patches(batch, x + 1, y, z)
        return others

    end_patch_index = int(n_patches_one_dim / 2) - 1  # this is the last index of the terms
    for batch_index in range(batch_size):
        for col_index in range(n_patches_one_dim):
            for depth_index in range(n_patches_one_dim):
                # positive example
                terms = get_patches_for(batch_index, end_patch_index, col_index, depth_index)
                predict_terms = get_following_patches(batch_index, end_patch_index + 2, col_index, depth_index)
                patches_enc.append(np.stack(terms))
                patches_pred.append(np.stack(predict_terms))
                labels.append(1)

                # negative example
                r_batch = batch_index
                r_col = col_index
                r_dep = depth_index

                while r_batch == batch_index and r_col == col_index and r_dep == depth_index:
                    r_batch = np.random.randint(batch_size)
                    r_col = np.random.randint(n_patches_one_dim)
                    r_dep = np.random.randint(n_patches_one_dim)

                predict_terms = get_following_patches(r_batch, end_patch_index + 2, r_col, r_dep)
                patches_enc.append(np.stack(terms))
                patches_pred.append(np.stack(predict_terms))
                labels.append(0)

    return [np.stack(patches_enc), np.stack(patches_pred)], np.array(labels)
