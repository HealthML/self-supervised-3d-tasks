import numpy as np
import tensorflow.keras as keras


class DataGeneratorUnlabeled3D(keras.utils.Sequence):

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 shuffle=True,
                 pre_proc_func=None,
                 data_dim=None):
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.shuffle = shuffle
        self.path_to_data = data_path
        self.pre_proc_func = pre_proc_func
        self.on_epoch_end()

        if data_dim is None:
            self.dim = None
        else:
            self.dim = (data_dim, data_dim, data_dim)

        assert len(file_list) > 0, "received no files"

        # check how many examples preprocess produces for one file
        self.index_multiplicator = len(self.__data_generation([file_list[0]])[0])

        assert self.index_multiplicator > 0, "invalid preprocessing"

        print(self.index_multiplicator)

        if self.shuffle:
            # shuffle the files
            np.random.shuffle(self.list_IDs)

    def __len__(self):
        return int(np.ceil((len(self.list_IDs) * self.index_multiplicator) / self.batch_size))

    def __getitem__(self, index):
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
        X, Y = self.__data_generation(list_files_temp)

        relative_end = relative_start + self.batch_size

        if relative_end > len(X):
            relative_end = None

        X = X[relative_start:relative_end]
        Y = Y[relative_start:relative_end]

        return (X, Y)

    def on_epoch_end(self):
        if self.shuffle:
            # shuffle the files
            np.random.shuffle(self.list_IDs)

    def __data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)

            img = np.load(path_to_image)

            if self.dim is not None:
                assert img.shape[0] == self.dim[0], "shapes should match dim, no resizing here!"

            data_x.append(img)
            data_y.append(0)  # just to keep the dims right

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        if self.pre_proc_func:
            data_x, data_y = self.pre_proc_func(data_x, data_y)

        return data_x, data_y
