import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import Sequence, to_categorical
from sklearn.utils import resample


class KaggleGenerator(Sequence):
    def __init__(
            self,
            csvDescriptor=Path("/mnt/mpws2019cl1/kaggle_retina/train/trainLabels_shuffled.csv"),
            base_path=Path("/mnt/mpws2019cl1/kaggle_retina/train/resized_384"),
            batch_size=1,
            label_column="level",
            resize_to=(384, 384),
            num_classes=5,
            split=False,
            shuffle=False,
            pre_proc_func_train=None,
            pre_proc_func_val=None,
            categorical = True,
            sample_classes_uniform = False,
            discard_part_of_dataset_split = False
    ):
        if discard_part_of_dataset_split > 1 or discard_part_of_dataset_split < -1:
            raise ValueError("cannot discard more than everything or less than nothing")

        if sample_classes_uniform and not shuffle:
            raise ValueError("shuffle and sample_classes_uniform have to be both active")

        self.categorical = categorical
        self.pre_proc_func_train = pre_proc_func_train
        self.pre_proc_func_val = pre_proc_func_val
        self.dataset = pd.read_csv(csvDescriptor)

        if discard_part_of_dataset_split:
            x_len = len(self.dataset)
            slice = int(x_len*discard_part_of_dataset_split)

            if slice < 0:
                self.dataset = self.dataset[slice:]
            else:
                self.dataset = self.dataset[:slice]

        if sample_classes_uniform:
            df_majority = self.dataset[self.dataset.level == 0]
            df_minorities = [self.dataset[self.dataset.level == i] for i in range(1, 5)]
            df_minorities = [resample(minority, replace=True, n_samples=len(df_majority))
                             for minority in df_minorities]

            self.dataset = pd.concat([df_majority] + df_minorities)

        if shuffle:
            self.dataset = self.dataset.sample(frac=1)

        self.batch_size = batch_size
        self.dataset_len = len(self.dataset.index)
        self.train_len = self.dataset_len

        self.split = split
        if self.split:
            splitpoint = math.floor(self.dataset_len * split)
            self.train_len = splitpoint
            self.offset = splitpoint

        self.n_batches = int(math.ceil(self.train_len / batch_size))
        self.label_column = label_column
        self.num_classes = num_classes
        self.base_path = Path(base_path)
        self.resize_width = resize_to[0] if resize_to[0] > 32 else 32
        self.resize_height = resize_to[1] if resize_to[1] > 32 else 32

    def __len__(self):
        return self.n_batches

    def load_image(self, index):
        path = self.base_path / self.dataset.iloc[index][0]
        path = path.with_suffix(".jpeg")
        image = Image.open(path)
        if image.width != self.resize_width or image.height != self.resize_height:
            image = image.resize(
                (self.resize_width, self.resize_height), resample=Image.LANCZOS
            )
        return np.array(image) / 255.0

    def get_val_data(self, debug=False):
        assert (
            self.split
        ), "To use Validation Data a fractional split has to be given initially."
        endpoint = self.dataset_len if not debug else self.offset + 200
        X_t = []
        Y_t = []
        for c in range(self.offset, endpoint):  # TODO: remove val set binding
            X_t.append(self.load_image(c))
            Y_t.append(self.dataset.iloc[c][self.label_column])

        data_x = np.array(X_t)

        if self.categorical:
            data_y = to_categorical(np.array(Y_t), num_classes=self.num_classes)
        else:
            data_y = np.array(Y_t)

        if self.pre_proc_func_val:
            data_x, data_y = self.pre_proc_func_val(data_x, data_y)

        return (
            data_x,
            data_y,
        )

    def __getitem__(self, index):
        X_t = []
        Y_t = []
        for c in range(index, min(index + self.batch_size, self.train_len)):
            X_t.append(self.load_image(c))
            Y_t.append(self.dataset.iloc[c][self.label_column])

        data_x = np.array(X_t)

        if self.categorical:
            data_y = to_categorical(np.array(Y_t), num_classes=self.num_classes)
        else:
            data_y = np.array(Y_t)

        if self.pre_proc_func_train:
            data_x, data_y = self.pre_proc_func_train(data_x, data_y)

        return (
            data_x,
            data_y,
        )