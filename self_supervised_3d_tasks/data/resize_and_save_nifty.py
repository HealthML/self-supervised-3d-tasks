import os
from pathlib import Path

import nibabel as nib
import skimage.transform as skTrans
from self_supervised_3d_tasks.data.nifti_utils import read_scan_find_bbox, norm
import traceback
import numpy as np

def data_generation():
    result_path = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_bbox"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesPt"

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    dim = (128, 128, 128)
    list_files_temp = os.listdir(path_to_data)

    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)

        try:
            img = nib.load(path_to_image).get_fdata().astype(np.float32)
            img, bb = read_scan_find_bbox(img, thresh=0.15)

            img = skTrans.resize(img, dim, order=1, preserve_range=True)
            result = norm(np.expand_dims(img, axis=3))

            file_name = file_name[:file_name.index('.')] + ".npy"
            np.save("{}/{}".format(result_path, file_name), result)

            perc = (float(i) * 100.0) / len(list_files_temp)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            print(e)
            continue

def data_generation_labels():
    result_path = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_bbox_labeled/img"
    result_path_labels = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_bbox_labeled/img_labels"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesTr"
    path_to_labels = "/mnt/mpws2019cl1/Task07_Pancreas/labelsTr"

    Path(result_path).mkdir(exist_ok=True, parents=True)
    Path(result_path_labels).mkdir(exist_ok=True, parents=True)

    dim = (128, 128, 128)
    list_files_temp = os.listdir(path_to_data)

    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)
        path_to_label = "{}/{}".format(path_to_labels, file_name)

        try:
            img = nib.load(path_to_image).get_fdata().astype(np.float32)
            label = nib.load(path_to_label).get_fdata().astype(np.float32)

            img, bb = read_scan_find_bbox(img, thresh=0.15)
            label = label[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

            img = skTrans.resize(img, dim, order=1, preserve_range=True)
            label = skTrans.resize(label, dim, order=1, preserve_range=True)

            result = norm(np.expand_dims(img, axis=3))
            label_result = np.expand_dims(label, axis=3)

            file_name = file_name[:file_name.index('.')] + ".npy"
            np.save("{}/{}".format(result_path, file_name), result)
            np.save("{}/{}".format(result_path_labels, file_name), label_result)

            perc = (float(i) * 100.0) / len(list_files_temp)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            traceback.print_tb(e.__traceback__)
            continue


if __name__ == "__main__":
    data_generation_labels()
