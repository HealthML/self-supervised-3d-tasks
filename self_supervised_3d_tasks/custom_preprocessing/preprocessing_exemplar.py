import numpy as np

def preprocessing_exemplar(x, y, process_3d = False, embedding_layers=10):
    def _distort_color(scan):
        """
        This function is based on the distort_color function from the tf implementation.
        :param scan: image as np.array
        :return: processed image as np.array
        """
        # adjust brightness
        max_delta = 32.0 / 255.0
        delta = np.random.uniform(-max_delta, max_delta)
        scan += delta

        # adjust contrast
        lower = 0.5
        upper = 1.5
        contrast_factor = np.random.uniform(lower, upper)
        scan_mean = np.mean(scan)
        scan = (contrast_factor * (scan - scan_mean)) + scan_mean
        return scan

    """
    This function preprocess a batch for relative patch location in a 2 dimensional space.
    :param x: array of images
    :param y: None
    :return: x as np.array of images with random rotations, y np.array with one-hot encoded label
    """
    # get batch size
    batch_size = len(y)
    y = np.zeros((batch_size, 3, embedding_layers))
    if process_3d:
        x_processed = np.empty(shape=(batch_size, 3, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]))
    else:
        x_processed = np.empty(shape=(batch_size, 3, x.shape[-3], x.shape[-2], x.shape[-1]))
    # init patch array
    if process_3d:
        triplet = np.empty(shape=(3, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]))
    else:
        triplet = np.empty(shape=(3, x.shape[-3], x.shape[-2], x.shape[-1]))
    # get negative examples
    random_images = x
    np.random.shuffle(random_images)
    # loop over all images with index and image
    for index, image in enumerate(x):
        # random transformation [0..1]
        random_lr_flip = np.random.randint(0, 2)
        random_ud_flip = np.random.randint(0, 2)
        distort_color = np.random.randint(0, 2)
        processed_image = image
        # flip up and down
        if random_ud_flip == 1:
            processed_image = np.flip(processed_image, 0)
        # flip left and right
        if random_lr_flip == 1:
            processed_image = np.flip(processed_image, 1)
        # distort_color
        if distort_color == 1:
            processed_image = _distort_color(processed_image)

        # Set Anchor Image
        triplet[0] = processed_image
        # Set Positiv Image
        triplet[1] = image
        # Set negativ Image
        negativ_image = random_images[index]
        triplet[2] = negativ_image
        x_processed[index] = triplet
    # return images and rotation
    return x_processed, y