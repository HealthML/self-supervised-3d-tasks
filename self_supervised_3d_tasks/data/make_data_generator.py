import os

from self_supervised_3d_tasks.data.segmentation_task_loader import SegmentationGenerator3D


def get_data_generators(data_path, data_generator, train_split=None, val_split=None,
                        train_data_generator_args={},
                        test_data_generator_args={},
                        val_data_generator_args={}, **kwargs):
    """
    This function generates the data generator for training, testing and optional validation.
    :param data_path: path to files
    :param data_generator: generator to use, first arguments must be data_path and files
    :param train_split: between 0 and 1, percentage of images used for training
    :param val_split: between 0 and 1, percentage of images used for test, None for no validation set
    :param shuffle_files:
    :param train_data_generator_args: Optional arguments for data generator
    :param test_data_generator_args: Optional arguments for data generator
    :param val_data_generator_args: Optional arguments for data generator
    :param data3d: 3d training
    :return: returns data generators
    """

    # List images in directory
    files = os.listdir(data_path)

    if val_split:
        assert train_split, "val split cannot be set without train split"

    # Validation set is needed
    if val_split:
        assert val_split + train_split <= 1., "Invalid arguments for splits: {}, {}".format(val_split, train_split)
        # Calculate splits
        train_split = int(len(files) * train_split)
        val_split = int(len(files) * val_split)

        # Create lists
        train = files[0:train_split]
        val = files[train_split:train_split + val_split]
        test = files[train_split + val_split:]

        # create generators
        train_data_generator = data_generator(data_path, train, **train_data_generator_args)
        val_data_generator = data_generator(data_path, val, **val_data_generator_args)

        if len(test) > 0:
            test_data_generator = data_generator(data_path, test, **test_data_generator_args)
            # Return generators
            return train_data_generator, val_data_generator, test_data_generator
        else:
            return train_data_generator, val_data_generator, None
    elif train_split:
        assert train_split <= 1., "Invalid arguments for split: {}".format(train_split)

        # Calculate split
        train_split = int(len(files) * train_split)

        # Create lists
        train = files[0:train_split]
        val = files[train_split:]

        # Create data generators
        train_data_generator = data_generator(data_path, train, **train_data_generator_args)
        val_data_generator = data_generator(data_path, val, **val_data_generator_args)

        # Return generators
        return train_data_generator, val_data_generator
    else:
        train_data_generator = data_generator(data_path, files, **train_data_generator_args)
        return train_data_generator
