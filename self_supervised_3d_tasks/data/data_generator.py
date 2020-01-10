import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import os


class DataGeneratorUnlabeled(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 dim=(32, 32, 32),
                 n_channels=1,
                 shuffle=True,
                 image_format="jpeg"):
        '''
            :param data_path: path to directory with images
            :param file_list: list of files in directory for this data generator
            :param batch_size: int batch size
            :param dim: tuple of ints as dimension
            :param n_channels: number of channels
            :param shuffle: flag indicates shuffle after epoch
            :param image_format: image format for loading (e.g. jpegs)
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.indexes = np.arange(len(self.list_IDs))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path_to_data = data_path
        self.image_format = image_format
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        :param index: index of batch
        :return: return list of file_names in this batch
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        data_x = np.empty((self.batch_size, *self.dim, self.n_channels))
        data_y = [0 * self.batch_size]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path_to_image = "{}/{}".format(self.path_to_data, ID)
            data_x[i,] = np.array(plt.imread(path_to_image, format=self.image_format))
        return data_x, data_y


class DataGeneratorLabeled(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 dim=(32, 32, 32),
                 n_channels=1,
                 shuffle=True,
                 image_format="jpeg"):
        '''
            :param data_path: path to directory with images
            :param file_list: list of files in directory for this data generator
            :param batch_size: int batch size
            :param dim: tuple of ints as dimension
            :param n_channels: number of channels
            :param shuffle: flag indicates shuffle after epoch
            :param image_format: image format for loading (e.g. jpegs)
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.indexes = np.arange(len(self.list_IDs))
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path_to_data = data_path
        self.image_format = image_format
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        :param index: index of batch
        :return: return list of file_names in this batch
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        data_x = np.empty((self.batch_size, *self.dim, self.n_channels))
        # TODO: Adjust labels
        data_y = [0 * self.batch_size]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            path_to_image = "{}/{}".format(self.path_to_data, ID)
            data_x[i,] = np.array(plt.imread(path_to_image, format=self.image_format))
        return data_x, data_y



def get_data_generators(data_path, train_split=.6, labeled = False, val_split=None, shuffle_files=True, data_generator_args={}):
    '''
    This function generates the data generator for training, testing and optional validation.
    :param data_path: path to files
    :param train_split: between 0 and 1, percentage of images used for training
    :param val_split: between 0 and 1, percentage of images used for test, None for no validation set
    :param shuffle_files:
    :param data_generator_args: Optional arguments for data generator
    :return: returns data generators
    '''

    # List images in directory
    files = os.listdir(data_path)

    # Shuffle files
    if shuffle_files:
        np.random.shuffle(files)

    # Validation set is needed
    if val_split:
        assert (val_split + train_split >= 1., "Invalid arguments for splits: {}, {}".format(val_split, train_split))
        # Calculate splits
        train_split = int(len(files) * train_split)
        val_split = int(len(files) * val_split)

        # Create lists
        train = files[0:train_split]
        val = files[train_split:train_split + val_split]
        test = files[train_split + val_split:]

        # Create data generators
        if labeled:
            train_data_generator = DataGeneratorLabeled(data_path, train, **data_generator_args)
            test_data_generator = DataGeneratorLabeled(data_path, test, **data_generator_args)
            val_data_generator = DataGeneratorLabeled(data_path, val, **data_generator_args)
        else:
            train_data_generator = DataGeneratorUnlabeled(data_path, train, **data_generator_args)
            test_data_generator = DataGeneratorUnlabeled(data_path, test, **data_generator_args)
            val_data_generator = DataGeneratorUnlabeled(data_path, val, **data_generator_args)
        # Return generators
        return train_data_generator, test_data_generator, val_data_generator
    else:
        assert (train_split >= 1., "Invalid arguments for split: {}".format(train_split))

        # Calculate split
        train_split = int(len(files) * train_split)

        # Create lists
        train = files[0:train_split]
        test = files[train_split:]

        # Create data generators
        if labeled:
            train_data_generator = DataGeneratorLabeled(data_path, train, **data_generator_args)
            test_data_generator = DataGeneratorLabeled(data_path, test, **data_generator_args)
        else:
            train_data_generator = DataGeneratorUnlabeled(data_path, train, **data_generator_args)
            test_data_generator = DataGeneratorUnlabeled(data_path, test, **data_generator_args)

        # Return generators
        return train_data_generator, test_data_generator
