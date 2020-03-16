import functools
import random

import numpy as np
import albumentations as ab
import scipy.ndimage as ndimage

from self_supervised_3d_tasks.preprocessing.utils.crop import crop_3d
from self_supervised_3d_tasks.preprocessing.utils.pad import pad_to_final_size_3d
from self_supervised_3d_tasks.data.preproc_negative_sampling import NegativeSamplingPreprocessing


def augment_exemplar_2d(image):
    return ab.Compose(
        [
            ab.RandomRotate90(p=1),
            ab.VerticalFlip(),
            ab.HorizontalFlip(),
            ab.RandomBrightnessContrast(p=1),
        ]
    )(image=image)["image"]

def augment_exemplar_3d(image):
    # prob to apply transforms
    alpha = 0.5
    beta = 0.5
    gamma = 0.15  # takes way too much time

    rotate_only_90 = 0.5

    def _distort_zoom(scan):
        scan_shape = scan.shape
        factor = 0.2
        zoom_factors = [np.random.uniform(1 - factor, 1 + factor) for _ in range(scan.ndim - 1)] + [1]
        scan = ndimage.zoom(scan, zoom_factors, mode="constant")
        scan = pad_to_final_size_3d(scan, scan_shape[0])
        scan = crop_3d(scan, True, scan_shape)
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
    if np.random.rand() < alpha:
        if np.random.rand() < rotate_only_90:
            processed_image = np.rot90(processed_image, k=np.random.randint(0, 4), axes=(0, 1))
        else:
            processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(0, 1), reshape=False)

    if np.random.rand() < alpha:
        if np.random.rand() < rotate_only_90:
            processed_image = np.rot90(processed_image, k=np.random.randint(0, 4), axes=(1, 2))
        else:
            processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(1, 2), reshape=False)

    if np.random.rand() < alpha:
        if np.random.rand() < rotate_only_90:
            processed_image = np.rot90(processed_image, k=np.random.randint(0, 4), axes=(0, 2))
        else:
            processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(0, 2), reshape=False)

    if np.random.rand() < beta:
        # color distortion
        processed_image = _distort_color(processed_image)
    if np.random.rand() < gamma:
        # zooming
        processed_image = _distort_zoom(processed_image)

    return processed_image


def make_derangement(indices):
    if len(indices) == 1:
        return indices
    for i in range(len(indices) - 1, 0, -1):
        j = random.randrange(i)  # 0 <= j <= i-1
        indices[j], indices[i] = indices[i], indices[j]
    return indices


def preprocessing_exemplar_training_neg_sampling(nsp, ids, x, y, process_3d):
    batch_size = len(y)
    x_processed = np.empty(shape=(batch_size, 3, *x.shape[1:]))
    triplet = np.empty(shape=(3, *x.shape[1:]))

    for i, image in enumerate(x):
        if process_3d:
            processed_image = augment_exemplar_3d(image)
        else:
            processed_image = augment_exemplar_2d(image)
        triplet[0] = processed_image  # augmented
        triplet[1] = image.copy()  # original (pos.)
        x, _ = nsp.draw_neg_sample([ids[i]])
        triplet[2] = x  # negative
        x_processed[i] = triplet.copy()

    return x_processed, y


def preprocessing_exemplar_training(x, y, process_3d):
    batch_size = len(y)
    x_processed = np.empty(shape=(batch_size, 3, *x.shape[1:]))
    triplet = np.empty(shape=(3, *x.shape[1:]))
    derangement = make_derangement(list(range(len(x))))
    random_shuffled = x.copy()[derangement]

    for i, image in enumerate(x):
        if process_3d:
            processed_image = augment_exemplar_3d(image)
        else:
            processed_image = augment_exemplar_2d(image)
        triplet[0] = processed_image  # augmented
        triplet[1] = image.copy()  # original (pos.)
        triplet[2] = random_shuffled[i].copy()  # negative
        x_processed[i] = triplet.copy()
    return x_processed, y

def get_exemplar_training_preprocessing(process_3d=False, sample_neg_examples_from="batch"):
    if sample_neg_examples_from == "dataset":
        pp_f = functools.partial(preprocessing_exemplar_training_neg_sampling, process_3d=process_3d)
        nsp = NegativeSamplingPreprocessing(pp_f)
        return nsp
    elif sample_neg_examples_from == "batch":
        return functools.partial(preprocessing_exemplar_training, process_3d=process_3d)
    else:
        raise ValueError(f"Value {sample_neg_examples_from} is invalid")