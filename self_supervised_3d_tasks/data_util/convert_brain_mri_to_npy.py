import glob
import os
import zipfile

import nibabel as nib
import numpy as np
import traceback


def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im


def read_scan(sbbox, nif_file, normalize=True):
    if normalize:
        image = norm(nif_file.get_fdata()[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]])
    else:
        image = nif_file.get_fdata()[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]]
    return image


source_path = "/mnt/30T/data/ukbiobank/original/imaging/brain_mri/"
destination_path = "/mnt/30T/data/ukbiobank/derived/imaging/brain_mri/"

t1_zip_files = sorted(glob.glob(source_path + "T1_structural_brain_mri/archive/**/" + "*.zip", recursive=True))
t2_flair_zip_files = sorted(
    glob.glob(source_path + "T2_FLAIR_structural_brain_mri/archive/**/" + "*.zip", recursive=True))

t2_flair_patient_ids = dict()
for filename in t2_flair_zip_files:
    id = os.path.basename(filename).split("_")[0]
    t2_flair_patient_ids[id] = filename
del t2_flair_zip_files

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
            np.save(os.path.join(destination_path, "T1", str(subject_id) + ".npy"), t1_scan)

            t2_archive = zipfile.ZipFile(t2_flair_patient_ids[subject_id], 'r')
            t2_extracted_path = t2_archive.extract('T2_FLAIR/T2_FLAIR.nii.gz')
            nif_file = nib.load(t2_extracted_path)
            t2_scan = nif_file.get_fdata()
            t2_scan[~t1_mask] = t2_scan.min()
            np.save(os.path.join(destination_path, "T2_FLAIR", str(subject_id) + ".npy"), t2_scan)
            del t1_scan, t2_scan
        except Exception:
            print(t1_zip_file)
            print(traceback.format_exc())

    if count % 100 == 0:
        print("Processed " + str(count) + " scans so far.")