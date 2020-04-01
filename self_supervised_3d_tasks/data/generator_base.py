import numpy as np
import random
import tensorflow.keras as keras

from self_supervised_3d_tasks.data.preproc_negative_sampling import NegativeSamplingPreprocessing


class DataGeneratorBase(keras.utils.Sequence):
    def __init__(self,
                 file_list,
                 batch_size,
                 shuffle,
                 pre_proc_func,
                 use_realistic_batch_size=True):
        super(DataGeneratorBase, self).__init__()

        self.use_realistic_batch_size = use_realistic_batch_size
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.shuffle = shuffle
        self.on_epoch_end()
        self.index_multiplicator = None
        self.pre_proc_func = pre_proc_func

        if isinstance(self.pre_proc_func, NegativeSamplingPreprocessing):
            def neg_sampling(positive_ids):
                neg_ids = [e for e in self.list_IDs if e not in positive_ids]
                idx = np.random.randint(len(neg_ids))
                x, y = self.data_generation([neg_ids[idx]])

                return x[0], y[0]

            self.pre_proc_func.set_negative_sampling(neg_sampling)

        assert len(file_list) > 0, "received no files"

    def get_multiplicator(self):
        # check how many examples preprocess produces for one file
        self.index_multiplicator = DataGeneratorBase.get_batch_size(self.__data_generation_intern([self.list_IDs[0]])[0])
        assert self.index_multiplicator > 0, "invalid preprocessing"

    def __len__(self):
        if not self.use_realistic_batch_size:
            return int(np.ceil(len(self.list_IDs) / self.batch_size))

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
        if not self.use_realistic_batch_size:
            index_start = index * self.batch_size  # inc
            index_end = (index + 1) * self.batch_size  # exc

            if index_end > len(self.list_IDs):
                # last batch
                index_end = len(self.list_IDs)

            list_files_temp = [self.list_IDs[k] for k in range(index_start, index_end)]
            X, Y = self.__data_generation_intern(list_files_temp)
            return X, Y

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
        X, Y = self.__data_generation_intern(list_files_temp)

        relative_end = relative_start + self.batch_size

        if relative_end > len(X):
            relative_end = None

        X = DataGeneratorBase.slice_input(X, relative_start, relative_end)
        Y = DataGeneratorBase.slice_input(Y, relative_start, relative_end)

        return X, Y

    def on_epoch_end(self):
        # TODO: see issue: https://github.com/tensorflow/tensorflow/issues/35911 -- in fixing
        super(DataGeneratorBase, self).on_epoch_end()
        if self.shuffle:
            # shuffle the files
            random.shuffle(self.list_IDs)

    def __data_generation_intern(self, list_files_temp):
        data_x, data_y = self.data_generation(list_files_temp)

        if self.pre_proc_func:
            if isinstance(self.pre_proc_func, NegativeSamplingPreprocessing):
                data_x, data_y = self.pre_proc_func.preprocess_function(list_files_temp, data_x, data_y)
            else:
                data_x, data_y = self.pre_proc_func(data_x, data_y)

        return data_x, data_y

    def data_generation(self, list_files_temp):
        raise NotImplementedError("should be implemented in subclass")
