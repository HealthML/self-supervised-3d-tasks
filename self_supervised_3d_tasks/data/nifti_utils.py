import numpy as np


def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im


def read_scan_find_bbox(image, normalize=True):
    st_x, en_x, st_y, en_y, st_z, en_z = 0, 0, 0, 0, 0, 0
    for x in range(image.shape[0]):
        if np.any(image[x, :, :]):
            st_x = x
            break
    for x in range(image.shape[0] - 1, -1, -1):
        if np.any(image[x, :, :]):
            en_x = x
            break
    for y in range(image.shape[1]):
        if np.any(image[:, y, :]):
            st_y = y
            break
    for y in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, y, :]):
            en_y = y
            break
    for z in range(image.shape[2]):
        if np.any(image[:, :, z]):
            st_z = z
            break
    for z in range(image.shape[2] - 1, -1, -1):
        if np.any(image[:, :, z]):
            en_z = z
            break
    image = image[st_x:en_x, st_y:en_y, st_z:en_z]
    if normalize:
        image = norm(image)
    nbbox = np.array([st_x, en_x, st_y, en_y, st_z, en_z]).astype(int)
    return image, nbbox