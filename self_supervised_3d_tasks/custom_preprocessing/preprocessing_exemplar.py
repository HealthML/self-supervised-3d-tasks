import random

import numpy as np
import albumentations as ab
import scipy
import scipy.ndimage as ndimage


def augment_exemplar_3d(image):
    def _distort_zoom(scan):
        # i dont work yet
        scan_shape = scan.shape
        factor = 0.2
        zoom_factors = [np.random.uniform(1-factor, 1+factor) for _ in range(scan.ndim)]
        scan = ndimage.zoom(scan, zoom_factors, mode="constant", cval=0)
        return scan

    def _distort_color(scan):
        """
        This function is based on the distort_color function from the tf implementation.
        :param scan: image as np.array
        :return: processed image as np.array
        """
        # adjust brightness
        max_delta = 0.125
        delta = np.random.uniform(-max_delta, max_delta)
        scan += delta

        # adjust contrast
        lower = 0.5
        upper = 1.5
        contrast_factor = np.random.uniform(lower, upper)
        scan_mean = np.mean(scan)
        scan = (contrast_factor * (scan - scan_mean)) + scan_mean
        return scan

    processed_image = image.copy()
    for i in range(3):
        if np.random.rand() < 0.5:
            processed_image = np.flip(processed_image, i)

    # make rotation arbitrary instead of multiples of 90deg
    processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(0, 1), reshape=False)
    processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(1, 2), reshape=False)
    processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(0, 2), reshape=False)

    if np.random.rand() < 0.5:
        # color distortion
        processed_image = _distort_color(processed_image)
    return processed_image


def make_derangement(indices):
    if len(indices) == 1:
        return indices
    for i in range(len(indices) - 1, 0, -1):
        j = random.randrange(i)  # 0 <= j <= i-1
        indices[j], indices[i] = indices[i], indices[j]
    return indices


def preprocessing_exemplar_training(x, y, process_3d=False):
    batch_size = len(y)
    x_processed = np.empty(shape=(batch_size, 3, *x.shape[1:]))
    triplet = np.empty(shape=(3, *x.shape[1:]))
    derangement = make_derangement(list(range(len(x))))
    random_shuffled = x.copy()[derangement]
    for i, image in enumerate(x):
        if process_3d:
            processed_image = augment_exemplar_3d(image)
        else:
            processed_image = ab.Compose(
                [
                    ab.RandomRotate90(p=1),
                    ab.VerticalFlip(),
                    ab.HorizontalFlip(),
                    ab.RandomBrightnessContrast(p=1),
                ]
            )(image=image)["image"]
        triplet[0] = processed_image  # augmented
        triplet[1] = image.copy()  # original (pos.)
        triplet[2] = random_shuffled[i].copy()  # negative
        x_processed[i] = triplet.copy()
    return x_processed, y
