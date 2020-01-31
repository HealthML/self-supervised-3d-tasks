import random
import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.crop import crop_patches


def preprocess_image(image, is_training, split_per_side, patch_jitter, permutations):
    label = random.randint(0, len(permutations) - 1)
    patches = crop_patches(image, is_training, split_per_side, patch_jitter)

    b = np.zeros((len(permutations),))
    b[label] = 1

    return np.array(patches)[np.array(permutations[label])], np.array(b)


def preprocess(batch, split_per_side, patch_jitter, permutations, is_training=True):
    xs = []
    ys = []

    for image in batch:
        x, y = preprocess_image(image, is_training, split_per_side, patch_jitter, permutations)
        xs.append(x)
        ys.append(y)

    xs = np.stack(xs)
    ys = np.stack(ys)

    return xs, ys
