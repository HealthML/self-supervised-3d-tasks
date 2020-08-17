import os

import numpy as np
from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase


class DataGeneratorUnlabeled3D(DataGeneratorBase):

    def __init__(self, data_path, file_list, batch_size=32, shuffle=True, pre_proc_func=None):
        self.path_to_data = data_path

        super().__init__(file_list, batch_size, shuffle, pre_proc_func)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)
            if os.path.isfile(path_to_image):
                img = np.load(path_to_image)
                img = (img - img.min()) / (img.max() - img.min())

                data_x.append(img)
                data_y.append(0)  # just to keep the dims right

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        return data_x, data_y
