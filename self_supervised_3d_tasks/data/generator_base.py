import numpy as np
import random
import tensorflow.keras as keras


class DataGeneratorBase(keras.utils.Sequence):
    def __init__(self,
                 file_list,
                 batch_size=32,
                 shuffle=True):
        super(DataGeneratorBase, self).__init__()
        
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.shuffle = shuffle
        self.on_epoch_end()
        self.index_multiplicator = None

        assert len(file_list) > 0, "received no files"

    def get_multiplicator(self):
        # check how many examples preprocess produces for one file
        self.index_multiplicator = DataGeneratorBase.get_batch_size(self.data_generation([self.list_IDs[0]])[0])
        assert self.index_multiplicator > 0, "invalid preprocessing"

    def __len__(self):
        if self.index_multiplicator is None:
            self.get_multiplicator()

        return int(np.ceil((len(self.list_IDs) * self.index_multiplicator) / self.batch_size))

    @staticmethod
    def get_batch_size(x):
        if isinstance(x, list):
            return len(x[0])
        else:
            return len(x)

    @staticmethod
    def slice_input(x, start, end):
        if isinstance(x, list):
            result = []
            for ar in x:
                result.append(ar[start:end])
            return result
        else:
            return x[start:end]

    def __getitem__(self, index):
        if self.index_multiplicator is None:
            self.get_multiplicator()

        index_start = index * self.batch_size  # inc
        index_end = (index + 1) * self.batch_size  # exc

        if index_end > len(self.list_IDs) * self.index_multiplicator:
            # last batch
            # exclusive index
            index_end = len(self.list_IDs) * self.index_multiplicator

        file_start = int(np.floor(index_start / self.index_multiplicator))
        file_end = int(np.floor((index_end - 1) / self.index_multiplicator))

        relative_start = index_start % self.index_multiplicator

        list_files_temp = [self.list_IDs[k] for k in range(file_start, file_end + 1)]
        X, Y = self.data_generation(list_files_temp)

        relative_end = relative_start + self.batch_size

        if relative_end > len(X):
            relative_end = None

        X = DataGeneratorBase.slice_input(X, relative_start, relative_end)
        Y = DataGeneratorBase.slice_input(Y, relative_start, relative_end)

        return (X, Y)

    def on_epoch_end(self):
        # TODO: see issue: https://github.com/tensorflow/tensorflow/issues/35911 -- in fixing
        if self.shuffle:
            # shuffle the files
            random.shuffle(self.list_IDs)
        else:
            pass

    def data_generation(self, list_files_temp):
        raise NotImplementedError("should be implemented in subclass")
