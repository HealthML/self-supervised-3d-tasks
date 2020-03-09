from pathlib import Path

import numpy as np

from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase

NUM_CLASSES = 4


class SegmentationGenerator3D(DataGeneratorBase):
    def __init__(
            self,
            data_path,
            file_list,
            batch_size=8,
            pre_proc_func=None,
            shuffle=False
    ):
        self.label_dir = data_path + "_labels"
        self.data_dir = data_path
        self.pre_proc_func = pre_proc_func

        super(SegmentationGenerator3D, self).__init__(file_list, batch_size, shuffle)

    def load_image(self, index):
        file_name = self.input_images[index]
        path = "{}/{}".format(self.data_dir, file_name)
        path_label = "{}/{}".format(self.label_dir, file_name)

        return np.load(path), np.load(path_label)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path = "{}/{}".format(self.data_dir, file_name)
            path_label = Path("{}/{}".format(self.label_dir, file_name))
            path_label = path_label.with_name(path_label.stem + "_label").with_suffix(path_label.suffix)

            img = np.load(path)
            img = (img - img.min()) / (img.max() - img.min())
            data_x.append(img)
            data_y.append(np.load(path_label))

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        if self.pre_proc_func:
            data_x, data_y = self.pre_proc_func(data_x, data_y)

        data_y = np.rint(data_y).astype(np.int)
        data_y = np.eye(NUM_CLASSES)[data_y]
        data_y = np.squeeze(data_y, axis=-2)  # remove second last axis, which is still 1

        return data_x, data_y
