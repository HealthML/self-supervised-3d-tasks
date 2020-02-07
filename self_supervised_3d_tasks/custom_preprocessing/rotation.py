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
        square_size = min(image.shape[0], image.shape[1])
        if image.shape[0] != image.shape[1]:
            square_size = min(image.shape[0], image.shape[1])
            image = ab.CenterCrop(height=square_size, width=square_size)(image=image)['image']
        # random transformation [0..3]
        rot = np.random.random_integers(4) - 1
        # iterate over rotations
        for i in range(0, rot):
            # rotate the image
            image = np.rot90(image)
        # set image
        rotated_batch.append(image)
        # set index
        y[index, rot] = 1
    # return images and rotation
    return np.array(rotated_batch), y


def resize(batch, new_size):
    return np.array([ab.Resize(new_size, new_size)(image=image)["image"] for image in batch])
