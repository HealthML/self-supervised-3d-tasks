import numpy as np
import albumentations as ab


def crop_patches3d(image, is_training, split_per_side, patch_jitter=0):
    h, w, d, _ = image.shape

    patch_overlap = -patch_jitter if patch_jitter < 0 else 0

    h_grid = (h - patch_overlap) // split_per_side
    w_grid = (w - patch_overlap) // split_per_side
    d_grid = (d - patch_overlap) // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    d_patch = d_grid - patch_jitter

    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            for k in range(split_per_side):

                p = do_crop_3d(image,
                            j * h_grid,
                            i * w_grid,
                            k * d_grid,
                            h_grid + patch_overlap,
                            w_grid + patch_overlap,
                            d_grid + patch_overlap)

                if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                    p = crop3d(p, is_training, [h_patch, w_patch, d_patch])

                patches.append(p)

    return patches


def crop_patches(image, is_training, split_per_side, patch_jitter=0):
    h, w, _ = image.shape

    patch_overlap = -patch_jitter if patch_jitter < 0 else 0

    h_grid = (h - patch_overlap) // split_per_side
    w_grid = (w - patch_overlap) // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter

    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):

            p = do_crop(image,
                        j * h_grid,
                        i * w_grid,
                        h_grid + patch_overlap,
                        w_grid + patch_overlap)

            if h_patch < h_grid or w_patch < w_grid:
                p = crop(p, is_training, [h_patch, w_patch])

            patches.append(p)

    return patches


def crop(image, is_training, crop_size):
    h, w, = crop_size[0], crop_size[1]
    # c = image.shape[2]
    # h_old, w_old = image.shape[0], image.shape[1]

    if is_training:
        return ab.RandomCrop(h, w)(image=image)["image"]

        # x = np.random.randint(0, h_old-h)
        # y = np.random.randint(0, w_old-w)
    else:
        return ab.CenterCrop(h, w)(image=image)["image"]

        # x = (h_old - h) / 2
        # y = (w_old - w) / 2

    # return do_crop(image, x, y, h, w)


def crop3d(image, is_training, crop_size):
    h, w, d = crop_size[0], crop_size[1], crop_size[2]
    h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]

    if is_training:
        # crop random
        x = np.random.randint(0, h_old-h)
        y = np.random.randint(0, w_old-w)
        z = np.random.randint(0, d_old-d)
    else:
        # crop center
        x = (h_old - h) / 2
        y = (w_old - w) / 2
        z = (d_old - d) / 2

    return do_crop_3d(image, x, y, z, h, w, d)


def do_crop(image, x, y, h, w):
    return ab.Crop(x, y, x + h, y + w)(image=image)["image"]
    # return image[x:x + h, y:y + w, :]


def do_crop_3d(image, x, y, z, h, w, d):
    return image[x:x + h, y:y + w, z:z + d, :]
