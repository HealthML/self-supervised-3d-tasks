import os
import random


def get_data_generators_internal(data_path, files, data_generator, train_split=None, val_split=None,
                                 train_data_generator_args={},
                                 test_data_generator_args={},
                                 val_data_generator_args={},
                                 **kwargs):
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

        if len(val) > 0:
            val_data_generator = data_generator(data_path, val, **val_data_generator_args)
            return train_data_generator, val_data_generator
        else:
            return train_data_generator, None
    else:
        train_data_generator = data_generator(data_path, files, **train_data_generator_args)
        return train_data_generator


class CrossValidationDataset():
    def __init__(self, chunks, data_path, data_generator, train_data_generator_args={},
                 test_data_generator_args={},
                 val_data_generator_args={}, **kwargs):

        self.k_fold = len(chunks)
        self.chunks = chunks
        self.data_path = data_path
        self.data_generator = data_generator
        self.kwargs = kwargs

        self.train_data_generator_args = train_data_generator_args
        self.test_data_generator_args = test_data_generator_args
        self.val_data_generator_args = val_data_generator_args

    def make_generators(self, test_chunk, train_split=None, val_split=None):
        test = self.chunks[test_chunk]
        train_val = []
        for i in range(self.k_fold):
            if not (i == test_chunk):
                train_val += self.chunks[i]

        train_and_val = get_data_generators_internal(self.data_path, train_val, self.data_generator,
                                                     train_split=train_split,
                                                     val_split=val_split,
                                                     # val split can only be used to throw away some data
                                                     train_data_generator_args=self.train_data_generator_args,
                                                     val_data_generator_args=self.val_data_generator_args,
                                                     test_data_generator_args=self.test_data_generator_args,
                                                     **self.kwargs)
        test = get_data_generators_internal(self.data_path, test, self.data_generator, train_split=None,
                                            val_split=None,
                                            train_data_generator_args=self.test_data_generator_args,
                                            **self.kwargs)

        if len(train_and_val) > 2:
            train_and_val = train_and_val[:2]  # remove the test generator

        return train_and_val + (test,)


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def make_cross_validation(data_path, data_generator, k_fold=5, files=None,
                          train_data_generator_args={},
                          test_data_generator_args={},
                          val_data_generator_args={},
                          shuffle_before_split=False,
                          **kwargs):
    if files is None:
        # List images in directory
        files = os.listdir(data_path)

    if shuffle_before_split:
        random.shuffle(files)

    chunks = chunkify(files, k_fold)
    return CrossValidationDataset(chunks, data_path, data_generator, train_data_generator_args,
                                  test_data_generator_args, val_data_generator_args, **kwargs)


def get_data_generators(data_path, data_generator, train_split=None, val_split=None,
                        train_data_generator_args={},
                        test_data_generator_args={},
                        val_data_generator_args={},
                        shuffle_before_split=False,
                        **kwargs):
    """
    This function generates the data generator for training, testing and optional validation.
    :param data_path: path to files
    :param data_generator: generator to use, first arguments must be data_path and files
    :param train_split: between 0 and 1, percentage of images used for training
    :param val_split: between 0 and 1, percentage of images used for test, None for no validation set
    :param shuffle_before_split:
    :param train_data_generator_args: Optional arguments for data generator
    :param test_data_generator_args: Optional arguments for data generator
    :param val_data_generator_args: Optional arguments for data generator
    :return: returns data generators
    """

    # List images in directory
    files = os.listdir(data_path)

    if shuffle_before_split:
        random.shuffle(files)

    return get_data_generators_internal(data_path, files, data_generator, train_split=train_split, val_split=val_split,
                                        train_data_generator_args=train_data_generator_args,
                                        test_data_generator_args=test_data_generator_args,
                                        val_data_generator_args=val_data_generator_args,
                                        **kwargs)
