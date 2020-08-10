from pathlib import Path

import numpy as np
from scipy import ndimage

from self_supervised_3d_tasks.data.generator_base import DataGeneratorBase


class SegmentationGenerator3D(DataGeneratorBase):
    def __init__(
            self,
            data_path,
            file_list,
            batch_size=8,
            pre_proc_func=None,
            shuffle=False,
            augment=False,
            label_stem="_label"
    ):
        self.augment_scans_train = augment

        self.label_stem = label_stem
        self.label_dir = data_path + "_labels"
        self.data_dir = data_path

        super(SegmentationGenerator3D, self).__init__(file_list, batch_size, shuffle, pre_proc_func)

    def load_image(self, index):
        file_name = self.input_images[index]
        path = "{}/{}".format(self.data_dir, file_name)
        path_label = "{}/{}".format(self.label_dir, file_name)

        return np.load(path), np.load(path_label)

    def augment_3d(self, x, y):
        def _distort_color(scan):
            # adjust brightness
            max_delta = 0.125
            delta = np.random.uniform(-max_delta, max_delta)
            scan += delta
            # adjust contrast
            lower = 0.5
            upper = 1.5
            contrast_factor = np.random.uniform(lower, upper)
            scan_mean = np.mean(scan)
            scan = (contrast_factor * (scan - scan_mean)) + scan_mean
            return scan

        processed_image, processed_mask = x.copy(), y.copy()
        for i in range(3):
            if np.random.rand() < 0.5:
                processed_image = np.flip(processed_image, i)
                processed_mask = np.flip(processed_mask, i)
        # make rotation arbitrary instead of multiples of 90deg
        if np.random.rand() < 0.5:
            axis_choice = np.random.randint(0, 3)
            if axis_choice == 0:
                xy_angle = np.random.uniform(0, 360)
                processed_image = ndimage.rotate(processed_image, xy_angle, axes=(0, 1), reshape=False, order=1)
                processed_mask = ndimage.rotate(processed_mask, xy_angle, axes=(0, 1), reshape=False, order=0)
            elif axis_choice == 1:
                yz_angle = np.random.uniform(0, 360)
                processed_image = ndimage.rotate(processed_image, yz_angle, axes=(1, 2), reshape=False, order=1)
                processed_mask = ndimage.rotate(processed_mask, yz_angle, axes=(1, 2), reshape=False, order=0)
            else:
                xz_angle = np.random.uniform(0, 360)
                processed_image = ndimage.rotate(processed_image, xz_angle, axes=(0, 2), reshape=False, order=1)
                processed_mask = ndimage.rotate(processed_mask, xz_angle, axes=(0, 2), reshape=False, order=0)
        if np.random.rand() < 0.7:
            # color distortion (THIS DOESN'T CHANGE IN THE MASK)
            processed_image = _distort_color(processed_image)
        return processed_image, processed_mask

    def data_generation(self, list_files_temp):
        data_x = []
        data_y = []

        for file_name in list_files_temp:
            path = "{}/{}".format(self.data_dir, file_name)
            path_label = Path("{}/{}".format(self.label_dir, file_name))
            path_label = path_label.with_name(path_label.stem + self.label_stem).with_suffix(path_label.suffix)

            mask = np.load(path_label)
            img = np.load(path)
            img = (img - img.min()) / (img.max() - img.min())
            if self.augment_scans_train:
                img, mask = self.augment_3d(img, mask)
            data_x.append(img)
            data_y.append(mask)

        data_x = np.stack(data_x)
        data_y = np.stack(data_y)

        data_y = np.rint(data_y).astype(np.int)
        n_classes = np.max(data_y) + 1
        data_y = np.eye(n_classes)[data_y]
        if data_y.shape[-2] == 1:
            data_y = np.squeeze(data_y, axis=-2)  # remove second last axis, which is still 1

        return data_x, data_y
