from PIL import Image
import numpy as np
from tensorflow.python.keras.preprocessing.image import random_zoom
import albumentations as ab
from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase


class DataGeneratorUnlabeled2D(DataGeneratorBase):

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 shuffle=False,
                 pre_proc_func=None,
                 augment=False,
                 augment_zoom_only=False):
        self.augment_zoom_only = augment_zoom_only
        self.augment = augment
        self.path_to_data = data_path

        super(DataGeneratorUnlabeled2D, self).__init__(file_list, batch_size, shuffle, pre_proc_func)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)

            try:
                im_frame = Image.open(path_to_image)
                img = np.asarray(im_frame, dtype="float32")
                img /= 255

                if self.augment_zoom_only:
                    img = random_zoom(img, zoom_range=(0.85, 1.15), channel_axis=2, row_axis=0, col_axis=1,
                                      fill_mode='constant', cval=0.0)
                elif self.augment:
                    img = random_zoom(img, zoom_range=(0.85, 1.15), channel_axis=2, row_axis=0, col_axis=1,
                                        fill_mode='constant', cval=0.0)
                    img = ab.HorizontalFlip()(image=img)["image"]
                    img = ab.VerticalFlip()(image=img)["image"]

                data_x.append(img)
                data_y.append(0)
            except:
                print("Error while loading image {}.".format(path_to_image))
                continue

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        return data_x, data_y
