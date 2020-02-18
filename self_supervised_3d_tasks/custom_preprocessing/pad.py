import numpy as np


def pad_to_final_size_3d(volume, w):
    dim = volume.shape[0]
    f1 = int((w - dim) / 2)
    f2 = (w - dim) - f1

    result = np.pad(volume, ((f1, f2), (f1, f2), (f1, f2), (0, 0)), "edge")

    return result