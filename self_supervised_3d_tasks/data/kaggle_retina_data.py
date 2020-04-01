from pathlib import Path

import albumentations as ab
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import resample
from tensorflow.python.keras.preprocessing.image import random_zoom

from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase
from self_supervised_3d_tasks.data.make_data_generator import get_data_generators_internal, make_cross_validation


class KaggleGenerator(DataGeneratorBase):
    def __init__(
            self,
            data_path,
            file_list,
            dataset_table,
            batch_size=8,
            shuffle=False,
            suffix=".jpeg",
            pre_proc_func=None,
            multilabel=False,
            augment=False):

        self.augment = augment
        self.multilabel = multilabel
        self.suffix = suffix
        self.dataset = dataset_table
        self.base_path = Path(data_path)

        super().__init__(file_list, batch_size, shuffle, pre_proc_func)

    def load_image(self, index):
        path = self.base_path / self.dataset.iloc[index][0]
        image = Image.open(path.with_suffix(self.suffix))

        arr = np.array(image, dtype="float32")
        arr = arr / 255.0
        return arr

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for c in list_files_temp:
            image = self.load_image(c)
            label = self.dataset.iloc[c][1]

            if self.augment:
                image = random_zoom(image, zoom_range=(0.85, 1.15), channel_axis=2, row_axis=0, col_axis=1, fill_mode='constant', cval=0.0)
                image = ab.HorizontalFlip()(image=image)["image"]
                image = ab.VerticalFlip()(image=image)["image"]

            if self.multilabel:
                if label == 0:
                    label = [1, 0, 0, 0, 0]
                elif label == 1:
                    label = [1, 1, 0, 0, 0]
                elif label == 2:
                    label = [1, 1, 1, 0, 0]
                elif label == 3:
                    label = [1, 1, 1, 1, 0]
                elif label == 4:
                    label = [1, 1, 1, 1, 1]
                label = np.array(label)

            data_x.append(image)
            data_y.append(label)

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        return data_x, data_y

def __prepare_dataset(csv_file, sample_classes_uniform, shuffle_before_split):
    dataset = pd.read_csv(csv_file)

    if sample_classes_uniform:
        df_majority = dataset[dataset.level == 0]
        df_minorities = [dataset[dataset.level == i] for i in range(1, 5)]
        df_minorities = [resample(minority, replace=True, n_samples=len(df_majority))
                         for minority in df_minorities]

        dataset = pd.concat([df_majority] + df_minorities)
        dataset = dataset.sample(frac=1)  # dont make them appear in order

    if shuffle_before_split:
        dataset = dataset.sample(frac=1)

    file_list = list(range(len(dataset)))
    return file_list, dataset


def get_kaggle_cross_validation(data_path, csv_file, sample_classes_uniform=False, k_fold=5,
                        train_data_generator_args={},
                        test_data_generator_args={},
                        val_data_generator_args={},
                        shuffle_before_split=False,
                        **kwargs):
    file_list, dataset = __prepare_dataset(csv_file, sample_classes_uniform, shuffle_before_split)

    return make_cross_validation(data_path, KaggleGenerator, k_fold=k_fold, files=file_list,
                            train_data_generator_args={**{"dataset_table": dataset}, **train_data_generator_args},
                            test_data_generator_args={**{"dataset_table": dataset}, **test_data_generator_args},
                            val_data_generator_args={**{"dataset_table": dataset}, **val_data_generator_args},
                            shuffle_before_split=False,  # dont shuffle again
                            **kwargs)

def get_kaggle_generator(data_path, csv_file, sample_classes_uniform=False, train_split=None, val_split=None, train_data_generator_args={},
                         test_data_generator_args={}, val_data_generator_args={}, shuffle_before_split=False, **kwargs):
    file_list, dataset = __prepare_dataset(csv_file, sample_classes_uniform, shuffle_before_split)

    return get_data_generators_internal(data_path, file_list, KaggleGenerator, train_split=train_split, val_split=val_split,
                        train_data_generator_args={**{"dataset_table": dataset}, **train_data_generator_args},
                        test_data_generator_args={**{"dataset_table": dataset}, **test_data_generator_args},
                        val_data_generator_args={**{"dataset_table": dataset}, **val_data_generator_args},
                        **kwargs)