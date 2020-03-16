import cv2
import numpy as np
import albumentations as ab


def equalize(image):
    image = ab.Equalize(always_apply=True)(image=image)["image"]
    return image


def normalize(image):
    m = np.mean(image, axis=(0, 1)) / 255.0
    sd = np.std(image, axis=(0, 1)) / 255.0

    image = ab.Normalize(mean=m, std=sd)(image=image)["image"]
    return image


def blur_and_subtract(image):
    # Blur the image
    blurred = cv2.blur(image, ksize=(40, 40))
    dst = cv2.addWeighted(image, 2, blurred, -2, 0.5)

    dst = (dst - dst.min()) / np.ptp(dst)  # normalize afterwards
    return dst


def apply_to_x(x,y,func=blur_and_subtract):
    result = []

    for i in x:
        result.append(func(i))

    return np.asarray(result),y