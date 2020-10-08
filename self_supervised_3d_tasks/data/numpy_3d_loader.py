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


class PatchDataGeneratorUnlabeled3D(DataGeneratorBase):

    def __init__(self, data_path, file_list, batch_size=32, patch_size=(128, 128, 128), patches_per_scan=2,
                 shuffle=True, pre_proc_func=None):
        self.path_to_data = data_path
        self.patch_size = patch_size
        self.patches_per_scan = patches_per_scan

        super().__init__(file_list, batch_size, shuffle, pre_proc_func)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)
            if os.path.isfile(path_to_image):
                img = np.load(path_to_image)
                img = (img - img.min()) / (img.max() - img.min())

                origin_row = np.random.randint(0, img.shape[0] - self.patch_size[0], self.patches_per_scan)
                origin_col = np.random.randint(0, img.shape[1] - self.patch_size[1], self.patches_per_scan)
                origin_dep = np.random.randint(0, img.shape[2] - self.patch_size[2], self.patches_per_scan)

                for o_r, o_c, o_d in zip(origin_row, origin_col, origin_dep):
                    data_x.append(
                        img[o_r:o_r + self.patch_size[0], o_c:o_c + self.patch_size[1], o_d:o_d + self.patch_size[2]])
                    data_y.append(0)  # just to keep the dims right

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        return data_x, data_y
