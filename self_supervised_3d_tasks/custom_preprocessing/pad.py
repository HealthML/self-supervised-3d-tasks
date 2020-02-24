import cv2
import numpy as np
import albumentations as ab


def pad_to_final_size_3d(volume, w):
    dim = volume.shape[0]
    f1 = int((w - dim) / 2)
    f2 = (w - dim) - f1

    result = np.pad(volume, ((f1, f2), (f1, f2), (f1, f2), (0, 0)), mode="constant", constant_values=0)

    return result


def pad_to_final_size_2d(image, w):
    return ab.PadIfNeeded(w, w, border_mode=cv2.BORDER_CONSTANT, value=0)(image=image)["image"]
