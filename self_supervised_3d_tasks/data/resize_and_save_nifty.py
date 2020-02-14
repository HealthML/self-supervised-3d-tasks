import os

import nibabel as nib
import skimage.transform as skTrans
from self_supervised_3d_tasks.data.nifti_loader import read_scan_find_bbox
import traceback
import numpy as np


def data_generation():
    result_path = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesPt"
    dim = (128, 128, 128)
    list_files_temp = os.listdir(path_to_data)

    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)
        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()

            img, _ = read_scan_find_bbox(img)
            img = skTrans.resize(img, dim, order=1, preserve_range=True)

            result = np.expand_dims(img, axis=3)

            file_name = file_name[:file_name.index('.')]
            np.save("{}/{}".format(result_path, file_name), result)

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            traceback.print_tb(e.__traceback__)
            continue


if __name__ == "__main__":
    data_generation()
