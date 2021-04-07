import numpy as np
import albumentations as ab


def rotate_batch(x, y=None):
    """
    This function preprocess a batch for relative patch location in a 2 dimensional space.
    :param x: array of images
    :param y: None
    :return: x as np.array of images with random rotations, y np.array with one-hot encoded label
    """
    # get batch size
    batch_size = x.shape[0]
    # init np array with zeros
    y = np.zeros((batch_size, 4))
    rotated_batch = []
    # loop over all images with index and image
    for index, image in enumerate(x):
        # square the image
        if image.shape[0] != image.shape[1]:
            square_size = min(image.shape[0], image.shape[1])
            image = ab.CenterCrop(height=square_size, width=square_size)(image=image)['image']
        # random transformation [0..3]
        rot = np.random.random_integers(4) - 1
        image = np.rot90(image, rot)
        # set image
        rotated_batch.append(image)
        # set index
        y[index, rot] = 1
    # return images and rotation
    return np.stack(rotated_batch), y


def rotate_batch_3d(x, y=None):
    batch_size = x.shape[0]
    y = np.zeros((batch_size, 10))
    rotated_batch = []
    for index, volume in enumerate(x):
        rot = np.random.random_integers(10) - 1

        if rot == 0:
            volume = volume
        elif rot == 1:
            volume = np.transpose(np.flip(volume, 1), (1, 0, 2, 3))  # 90 deg Z
        elif rot == 2:
            volume = np.flip(volume, (0, 1))  # 180 degrees on z axis
        elif rot == 3:
            volume = np.flip(np.transpose(volume, (1, 0, 2, 3)), 1)  # 90 deg Z
        elif rot == 4:
            volume = np.transpose(np.flip(volume, 1), (0, 2, 1, 3))  # 90 deg X
        elif rot == 5:
            volume = np.flip(volume, (1, 2))  # 180 degrees on x axis
        elif rot == 6:
            volume = np.flip(np.transpose(volume, (0, 2, 1, 3)), 1)  # 90 deg X
        elif rot == 7:
            volume = np.transpose(np.flip(volume, 0), (2, 1, 0, 3))  # 90 deg Y
        elif rot == 8:
            volume = np.flip(volume, (0, 2))  # 180 degrees on y axis
        elif rot == 9:
            volume = np.flip(np.transpose(volume, (2, 1, 0, 3)), 0)  # 90 deg Y

        rotated_batch.append(volume)
        y[index, rot] = 1
    return np.stack(rotated_batch), y


def resize(batch, new_size):
    return np.array([ab.Resize(new_size, new_size)(image=image)["image"] for image in batch])
