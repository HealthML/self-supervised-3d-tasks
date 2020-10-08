import glob
import multiprocessing
import os
import traceback
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import skimage.transform as skTrans
from joblib import Parallel, delayed

from self_supervised_3d_tasks.data_util.nifti_utils import read_scan_find_bbox


def split_slices_to_single_files():
    result_path = "/mnt/mpws2019cl1/pancreas_data/img_slices_128_single"
    path_to_data = "/mnt/mpws2019cl1/pancreas_data/img_slices_128"

    Path(result_path).mkdir(parents=True, exist_ok=True)
    result_index = 0

    list_files_temp = list(Path(path_to_data).glob("*.npy"))
    for x, file_name in enumerate(list_files_temp):
        # load 3d scans
        source = np.load(str(file_name))

        for i in range(source.shape[2]):
            slice = source[:, :, i, :]
            np.save("{}/{}".format(result_path, "slice" + str(result_index) + ".npy"), slice)
            result_index += 1

        perc = (float(x) * 100.0) / len(list_files_temp)
        print(f"{perc:.2f} % done")


def data_generation_self_supervised_pancreas_2D_slices():
    result_path = "/home/Shared.Workspace/pancreas_data/images_slices_128_labeled"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesPt"

    dim_2d = (128, 128)
    list_files_temp = os.listdir(path_to_data)

    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)

        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()

            img, bb = read_scan_find_bbox(img)
            dim = dim_2d + (img.shape[2],)
            img = skTrans.resize(img, dim, order=1, preserve_range=True)

            result = np.expand_dims(img, axis=3)

            file_name = file_name[:file_name.index('.')] + ".npy"
            np.save("{}/{}".format(result_path, file_name), result)

            perc = (float(i) * 100.0) / len(list_files_temp)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            traceback.print_tb(e.__traceback__)
            continue


def data_generation_pancreas_2D_slices():
    result_path = "/home/Shared.Workspace/pancreas_data/images_slices_128_labeled/img"
    result_path_label = "/home/Shared.Workspace/pancreas_data/images_slices_128_labeled/img_label"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesTr"
    path_to_labels = "/mnt/mpws2019cl1/Task07_Pancreas/labelsTr"

    dim_2d = (128, 128)
    list_files_temp = os.listdir(path_to_data)

    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(result_path_label).mkdir(parents=True, exist_ok=True)

    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)
        path_to_label = "{}/{}".format(path_to_labels, file_name)

        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()

            label = nib.load(path_to_label)
            label = label.get_fdata()

            img, bb = read_scan_find_bbox(img)
            label = label[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

            dim = dim_2d + (img.shape[2],)
            img = skTrans.resize(img, dim, order=1, preserve_range=True)
            label = skTrans.resize(label, dim, order=1, preserve_range=True)

            result = np.expand_dims(img, axis=3)
            label_result = np.expand_dims(label, axis=3)

            file_name = file_name[:file_name.index('.')] + ".npy"
            np.save("{}/{}".format(result_path, file_name), result)
            np.save("{}/{}".format(result_path_label, file_name), label_result)

            perc = (float(i) * 100.0) / len(list_files_temp)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            print(e)
            traceback.print_tb(e.__traceback__)
            continue


def data_generation_pancreas():
    result_path = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_labeled"
    path_to_data = "/mnt/mpws2019cl1/Task07_Pancreas/imagesTr"
    path_to_labels = "/mnt/mpws2019cl1/Task07_Pancreas/labelsTr"

    dim = (128, 128, 128)
    list_files_temp = os.listdir(path_to_data)

    for i, file_name in enumerate(list_files_temp):
        path_to_image = "{}/{}".format(path_to_data, file_name)
        path_to_label = "{}/{}".format(path_to_labels, file_name)

        try:
            img = nib.load(path_to_image)
            img = img.get_fdata()

            label = nib.load(path_to_label)
            label = label.get_fdata()

            img, bb = read_scan_find_bbox(img)
            label = label[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

            img = skTrans.resize(img, dim, order=1, preserve_range=True)
            label = skTrans.resize(label, dim, order=1, preserve_range=True)

            result = np.expand_dims(img, axis=3)
            label_result = np.expand_dims(label, axis=3)

            file_name = file_name[:file_name.index('.')] + ".npy"
            label_file_name = file_name[:file_name.index('.')] + "_label.npy"
            np.save("{}/{}".format(result_path, file_name), result)
            np.save("{}/{}".format(result_path, label_file_name), label_result)

            perc = (float(i) * 100.0) / len(list_files_temp)
            print(f"{perc:.2f} % done")

        except Exception as e:
            print("Error while loading image {}.".format(path_to_image))
            traceback.print_tb(e.__traceback__)
            continue


def crop_one_volume(volume, volume_size, volume_for_resize=None):
    """
    get the bounding box of the non-zero region of an ND volume, crop/extract a subregion form an nd image.
    """
    if volume_for_resize is not None:
        input_shape = volume_for_resize.shape
        indxes = np.nonzero(volume_for_resize)
    else:
        input_shape = volume.shape
        indxes = np.nonzero(volume)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())
    # print('----',idx_max,idx_min)
    for i in range(len(input_shape)):
        difference = idx_max[i] - idx_min[i] - volume_size[i]
        idx_min[i] = int(idx_min[i] + abs(difference) / 2 - 1) if difference > 0 else \
            max(int(idx_min[i] - abs(difference) / 2 - 1), 0)
        idx_max[i] = idx_min[i] + volume_size[i]

    output = volume[..., :][np.ix_(range(idx_min[0], idx_max[0]),
                                   range(idx_min[1], idx_max[1]),
                                   range(idx_min[2], idx_max[2]))]
    return output


def data_conversion_ukb():
    source_path = "/mnt/projects/ukbiobank/original/imaging/brain_mri/"
    destination_path = "/mnt/projects/ukbiobank/derived/imaging/brain_mri/"
    input_volume_size = [160, 192, 192]

    t1_zip_files = sorted(glob.glob(source_path + "T1_structural_brain_mri/archive/**/*.zip", recursive=True))
    t2_zip_files = sorted(glob.glob(source_path + "T2_FLAIR_structural_brain_mri/archive/**/*.zip", recursive=True))

    t2_flair_patient_ids = dict()
    for filename in t2_zip_files:
        id = os.path.basename(filename).split("_")[0]
        t2_flair_patient_ids[id] = filename
    del t2_zip_files

    count = 0
    for t1_zip_file in t1_zip_files:
        subject_id = os.path.basename(t1_zip_file).split("_")[0]
        if subject_id in t2_flair_patient_ids:  # ensure we have a scan in t2 too
            count += 1
            try:
                t1_archive = zipfile.ZipFile(t1_zip_file, 'r')
                t1_extracted_path = t1_archive.extract('T1/T1.nii.gz')
                nif_file = nib.load(t1_extracted_path)
                t1_scan = nif_file.get_fdata()
                t1_extracted_path = t1_archive.extract('T1/T1_brain_mask.nii.gz')
                nif_file = nib.load(t1_extracted_path)
                t1_mask = np.asarray(nif_file.get_fdata(), dtype=np.bool)
                t1_scan[~t1_mask] = t1_scan.min()
                t1_scan[t1_scan < 0] = 0
                t1_scan[t1_scan > 4000] = 4000
                try:
                    t1_scan = crop_one_volume(t1_scan, input_volume_size)
                except:
                    t1_scan = skTrans.resize(t1_scan, input_volume_size, order=1, preserve_range=True)
                np.save(os.path.join(destination_path, "T1", str(subject_id) + ".npy"), t1_scan)

                t2_archive = zipfile.ZipFile(t2_flair_patient_ids[subject_id], 'r')
                t2_extracted_path = t2_archive.extract('T2_FLAIR/T2_FLAIR.nii.gz')
                nif_file = nib.load(t2_extracted_path)
                t2_scan = nif_file.get_fdata()
                t2_scan[~t1_mask] = t2_scan.min()
                t2_scan[t2_scan < 0] = 0
                t2_scan[t2_scan > 4000] = 4000
                try:
                    t2_scan = crop_one_volume(t2_scan, input_volume_size)
                except:
                    t2_scan = skTrans.resize(t2_scan, input_volume_size, order=1, preserve_range=True)
                np.save(os.path.join(destination_path, "T2_FLAIR", str(subject_id) + ".npy"), t2_scan)
                del t1_scan, t2_scan
            except Exception:
                print(t1_zip_file)
                print(traceback.format_exc())

        if count % 100 == 0:
            print("Processed " + str(count) + " scans so far.")


def data_conversion_ukb_masks():
    source_path = "/mnt/projects/ukbiobank/original/imaging/brain_mri/"
    destination_path = "/mnt/projects/ukbiobank/derived/imaging/brain_mri/"
    input_volume_size = [160, 192, 192]

    t1_zip_files = sorted(glob.glob(source_path + "T1_structural_brain_mri/archive/**/*.zip", recursive=True))
    t2_zip_files = sorted(glob.glob(source_path + "T2_FLAIR_structural_brain_mri/archive/**/*.zip", recursive=True))

    t2_flair_patient_ids = dict()
    for filename in t2_zip_files:
        id = os.path.basename(filename).split("_")[0]
        t2_flair_patient_ids[id] = filename
    del t2_zip_files

    count = 0
    for t1_zip_file in t1_zip_files:
        subject_id = os.path.basename(t1_zip_file).split("_")[0]
        if subject_id in t2_flair_patient_ids:  # ensure we have a scan in t2 too
            count += 1
            try:
                t1_archive = zipfile.ZipFile(t1_zip_file, 'r')
                t1_extracted_path = t1_archive.extract('T1/T1.nii.gz')
                nif_file = nib.load(t1_extracted_path)
                t1_scan = nif_file.get_fdata()
                t1_extracted_path = t1_archive.extract('T1/T1_brain_mask.nii.gz')
                nif_file = nib.load(t1_extracted_path)
                t1_mask = np.asarray(nif_file.get_fdata(), dtype=np.bool)
                try:
                    t1_mask = crop_one_volume(t1_mask, input_volume_size, volume_for_resize=t1_scan)
                except:
                    t1_mask = skTrans.resize(t1_mask, input_volume_size, order=1, preserve_range=True)
                np.save(os.path.join(destination_path, "MASKS", str(subject_id) + ".npy"), t1_mask)

                del t1_scan
            except Exception:
                print(t1_zip_file)
                print(traceback.format_exc())

        if count % 100 == 0:
            print("Processed " + str(count) + " scans so far.")


def data_conversion_brats(split='train'):
    """
    :param split: can be 'train' or 'val'
    """
    new_resolution = (192, 192, 160)
    train_path = '/mnt/dsets/brats/train/**/'
    validation_path = '/mnt/dsets/brats/test/**/'
    result_path = "/home/Aiham.Taleb/netstore/brats/images_resized"
    if split == 'train':
        path = train_path
    else:
        path = validation_path

    # loading the training images
    t1ce_files = sorted(glob.glob(path + "*_t1ce.nii.gz", recursive=True))
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    t1_files = sorted(glob.glob(path + "*_t1.nii.gz", recursive=True))
    t2_files = sorted(glob.glob(path + "*_t2.nii.gz", recursive=True))
    # loading the training labels (the segmentation masks)
    seg_files = sorted(glob.glob(path + "*_seg.nii.gz", recursive=True))
    print("reading scans...")
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(read_mm_slice_brats)(flair_files, i, seg_files, t1_files, t1ce_files, t2_files, new_resolution) for i in
        range(len(t1_files)))
    print("saving files..")
    for i, item in enumerate(results):
        file_name = os.path.basename(t1ce_files[i]).replace('_t1ce.nii.gz', '')
        scan_file_name = file_name + ".npy"
        mask_file_name = file_name + "_label.npy"
        np.save("{}/{}".format(result_path, scan_file_name), item[0])
        np.save("{}/{}".format(result_path, mask_file_name), item[1])

        perc = (float(i) * 100.0) / len(results)
        print(f"{perc:.2f} % done")
    print("Done.")


def read_mm_slice_brats(flair_files, i, seg_files, t1_files, t1ce_files, t2_files, new_resolution):
    t1ce_image = nib.load(t1ce_files[i]).get_fdata()
    t1ce_copy = np.copy(t1ce_image)
    try:
        t1ce_image = crop_one_volume(t1ce_image, new_resolution)
    except:
        t1ce_image = skTrans.resize(t1ce_image, new_resolution, order=1, preserve_range=True)
    flair_image = nib.load(flair_files[i]).get_fdata()
    try:
        flair_image = crop_one_volume(flair_image, new_resolution, volume_for_resize=t1ce_copy)
    except:
        flair_image = skTrans.resize(flair_image, new_resolution, order=1, preserve_range=True)
    t1_image = nib.load(t1_files[i]).get_fdata()
    try:
        t1_image = crop_one_volume(t1_image, new_resolution, volume_for_resize=t1ce_copy)
    except:
        t1_image = skTrans.resize(t1_image, new_resolution, order=1, preserve_range=True)
    t2_image = nib.load(t2_files[i]).get_fdata()
    try:
        t2_image = crop_one_volume(t2_image, new_resolution, volume_for_resize=t1ce_copy)
    except:
        t2_image = skTrans.resize(t2_image, new_resolution, order=1, preserve_range=True)
    seg_image = nib.load(seg_files[i]).get_fdata()
    try:
        seg_image = crop_one_volume(seg_image, new_resolution, volume_for_resize=t1ce_copy)
    except:
        seg_image = skTrans.resize(seg_image, new_resolution, order=0, preserve_range=True)
    seg_image = np.asarray(seg_image, dtype=np.int32)
    seg_image[seg_image == 4] = 3
    seg_image = np.expand_dims(seg_image, axis=-1)
    return np.stack([t1ce_image, flair_image, t1_image, t2_image], axis=-1), seg_image


def preprocess_ukb_3D_multimodal():
    base_path = "/mnt/30T/ukbiobank/derived/imaging/brain_mri/"
    result_path = "/mnt/30T/ukbiobank/derived/imaging/brain_mri/images_resized_128"
    t1_files = np.array(sorted(glob.glob(base_path + "/T1/**/*.npy", recursive=True)))
    t2_flair_files = np.array(sorted(glob.glob(base_path + "/T2_FLAIR/**/*.npy", recursive=True)))

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(read_ukb_scan_multimodal)(t1_files, t2_flair_files, i, result_path) for i in
        range(len(t2_flair_files)))
    print("done preprocessing images")


def read_ukb_scan_multimodal(t1_files, t2_flair_files, i, result_path):
    t1_scan, sbbox = read_scan_find_bbox(np.load(t1_files[i]))
    t2_flair_scan = np.load(t2_flair_files[i])[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]]
    stacked_array = np.stack([t1_scan, t2_flair_scan], axis=-1)
    scan_file_name = os.path.basename(t1_files[i])
    np.save("{}/{}".format(result_path, scan_file_name), stacked_array)
    perc = (float(i) * 100.0) / len(t2_flair_files)
    print(f"{perc:.2f} % done")


def resize_ukb_mask(mask_files, i, result_path):
    new_resolution = (128, 128, 128)
    mask, sbbox = read_scan_find_bbox(np.load(mask_files[i]))
    mask = skTrans.resize(mask, new_resolution, order=0, preserve_range=True)
    mask_file_name = os.path.basename(mask_files[i])
    np.save("{}/{}".format(result_path, mask_file_name), mask)
    perc = (float(i) * 100.0) / len(mask_files)
    print(f"{perc:.2f} % done")


def resize_ukb_3D_masks():
    base_path = "/mnt/projects/ukbiobank/derived/imaging/brain_mri/images_resized_128_labeled/test_labels"
    result_path = "/mnt/projects/ukbiobank/derived/imaging/brain_mri/images_resized_128_labeled/test_labels"
    mask_files = np.array(sorted(glob.glob(base_path + "/**/*.npy", recursive=True)))

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(resize_ukb_mask)(mask_files, i, result_path) for i in
        range(len(mask_files)))
    print("done preprocessing masks.")


def stack_ukb_scan_multimodal(t1_files, t2_flair_files, i, result_path):
    t1_scan = np.load(t1_files[i])
    t2_scan = np.load(t2_flair_files[i])
    stacked_array = np.stack([t1_scan, t2_scan], axis=-1)
    scan_file_name = os.path.basename(t1_files[i])
    np.save("{}/{}".format(result_path, scan_file_name), stacked_array)
    perc = (float(i) * 100.0) / len(t2_flair_files)
    print(f"{perc:.2f} % done")


def stack_ukb_3D_modalities():
    base_path = "/mnt/projects/ukbiobank/derived/imaging/brain_mri"
    result_path = "/mnt/projects/ukbiobank/derived/imaging/brain_mri/images_multimodal_resized"
    t1_files = np.array(sorted(glob.glob(base_path + "/T1/**/*.npy", recursive=True)))
    t2_flair_files = np.array(sorted(glob.glob(base_path + "/T2_FLAIR/**/*.npy", recursive=True)))

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(stack_ukb_scan_multimodal)(t1_files, t2_flair_files, i, result_path) for i in
        range(len(t2_flair_files)))
    print("done preprocessing images")


if __name__ == "__main__":
    # data_conversion_ukb()
    # data_conversion_ukb_masks()
    # stack_ukb_3D_modalities()
    data_conversion_brats(split='train')
    # data_conversion_brats(split='test')
