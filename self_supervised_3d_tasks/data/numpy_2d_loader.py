from pathlib import Path

from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase
import numpy as np

class DataGeneratorUnlabeled2D(DataGeneratorBase):

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 shuffle=False,
                 pre_proc_func=None):
        self.path_to_data = data_path
        self.label_dir = data_path + "_labels"

        if not Path(self.label_dir).exists():
            self.label_dir = None

        super(DataGeneratorUnlabeled2D, self).__init__(file_list, batch_size, shuffle, pre_proc_func)

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path_to_image = "{}/{}".format(self.path_to_data, file_name)

            try:
                mask = 0
                if self.label_dir:
                    path_label = Path("{}/{}".format(self.label_dir, file_name))
                    path_label = path_label.with_name(path_label.stem).with_suffix(path_label.suffix)
                    mask = np.load(path_label)

                path_to_image = "{}/{}".format(self.path_to_data, file_name)
                img = np.load(path_to_image)
                img = (img - img.min()) / (img.max() - img.min())

                for z in img.shape[2]:
                    data_x.append(img[:,:,z,0])

                    if mask:
                        data_y.append(mask[:,:,z,0])
                    else:
                        data_y.append(0)
            except:
                print("Error while loading image {}.".format(path_to_image))
                continue

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        return data_x, data_y