from PIL import Image
import numpy as np
from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase


class DataGeneratorUnlabeled2D(DataGeneratorBase):

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 shuffle=True,
                 pre_proc_func=None):
        self.path_to_data = data_path
        self.pre_proc_func = pre_proc_func

        super().__init__(file_list, batch_size, shuffle)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)

            try:
                im_frame = Image.open(path_to_image)

                img = np.asarray(im_frame, dtype="float32")
                img /= 255
                data_x.append(img)
                data_y.append(0)
            except:
                print("Error while loading image {}.".format(path_to_image))
                continue

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        if self.pre_proc_func:
            data_x, data_y = self.pre_proc_func(data_x, data_y)

        return data_x, data_y
