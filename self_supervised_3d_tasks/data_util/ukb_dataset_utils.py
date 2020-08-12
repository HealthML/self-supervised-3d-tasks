import glob
import multiprocessing
import random

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import skimage.transform as skTrans
# import skimage.transform as skTrans
import tensorflow as tf
import tensorflow.keras as keras
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder

SHARD_SIZE = 1024

resolution2D = (128, 128)
resolution3D = (128, 128, 128)


def parallel_load_ukb_multimodal(t1_files, t2_flair_files):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(read_ukb_scan_multimodal)(t1_files, t2_flair_files, i, resize=True) for i in
        range(len(t2_flair_files)))
    print("done loading images, gathering them in one array now.")
    all_slices = list()
    for mm_scan in results:
        if mm_scan[0].shape[2] == mm_scan[1].shape[2]:
            for z in range(mm_scan[0].shape[2]):
                t1_image = mm_scan[0][:, :, z]
                t2_flair_image = mm_scan[1][:, :, z]
                stacked_array = np.stack([t1_image, t2_flair_image], axis=-1)
                all_slices.append(stacked_array)

    return np.array(all_slices)


def parallel_load_ukb_3D_multimodal(t1_files, t2_flair_files):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(read_ukb_scan_multimodal)(t1_files, t2_flair_files, i, resize=False) for i in
        range(len(t2_flair_files)))
    print("done loading images, gathering them in one array now.")
    all_scans = list()
    for mm_scan in results:
        t1_image = mm_scan[0]
        t2_flair_image = mm_scan[1]
        stacked_array = np.stack([t1_image, t2_flair_image], axis=-1)
        all_scans.append(stacked_array)

    return np.array(all_scans)


def read_ukb_scan_multimodal(t1_files, t2_flair_files, i, resize):
    t1_scan, sbbox = read_scan_find_bbox(np.load(t1_files[i]), resize=resize)
    if not resize:
        t1_scan = skTrans.resize(t1_scan, resolution3D, order=1, preserve_range=True)
    t2_flair_scan = read_scan(sbbox, np.load(t2_flair_files[i]), resize=resize)
    if not resize:
        t2_flair_scan = skTrans.resize(t2_flair_scan, resolution3D, order=1, preserve_range=True)
    return t1_scan, t2_flair_scan


def read_scan_find_bbox(image, resize, normalize=True):
    if normalize:
        image = norm(image)
    st_x, en_x, st_y, en_y, st_z, en_z = 0, 0, 0, 0, 0, 0
    for x in range(image.shape[0]):
        if np.any(image[x, :, :]):
            st_x = x
            break
    for x in range(image.shape[0] - 1, -1, -1):
        if np.any(image[x, :, :]):
            en_x = x
            break
    for y in range(image.shape[1]):
        if np.any(image[:, y, :]):
            st_y = y
            break
    for y in range(image.shape[1] - 1, -1, -1):
        if np.any(image[:, y, :]):
            en_y = y
            break
    for z in range(image.shape[2]):
        if np.any(image[:, :, z]):
            st_z = z
            break
    for z in range(image.shape[2] - 1, -1, -1):
        if np.any(image[:, :, z]):
            en_z = z
            break
    image = image[st_x:en_x, st_y:en_y, st_z:en_z]
    if resize:
        new_image = np.zeros((resolution2D[0], resolution2D[1], image.shape[2]))
        for z in range(image.shape[2]):
            new_image[:, :, z] = cv2.resize(image[:, :, z], dsize=resolution2D)
    else:
        new_image = image
    nbbox = np.array([st_x, en_x, st_y, en_y, st_z, en_z]).astype(int)
    return new_image, nbbox


def read_scan(sbbox, image, resize, normalize=True):
    if normalize:
        image = norm(image[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]])
    else:
        image = image[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]]
    if resize:
        new_image = np.zeros((resolution2D[0], resolution2D[1], image.shape[2]))
        for z in range(image.shape[2]):
            new_image[:, :, z] = cv2.resize(image[:, :, z], dsize=resolution2D)
        return new_image
    else:
        return image


def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example(image_buffer, height, width, depth=None):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    channels = 2

    if depth is not None:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/depth': _int64_feature(depth),
            'image/channels': _int64_feature(channels),
            'image/encoded': _float_feature(image_buffer)}))
        return example
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/channels': _int64_feature(channels),
            'image/encoded': _float_feature(image_buffer)}))
        return example

if __name__ == "__main__":
    #################################
    ##      Test and Use Cases     ##
    #################################
    ukb_path = "/mnt/30T/ukbiobank/derived/imaging/brain_mri/"
    output_tf_records_path = "/mnt/30T/ukbiobank/derived/imaging/brain_mri/tf_records/"

    t1_files = np.array(sorted(glob.glob(ukb_path + "/T1/**/*.npy", recursive=True)))
    t2_flair_files = np.array(sorted(glob.glob(ukb_path + "/T2_FLAIR/**/*.npy", recursive=True)))
    num_files = t2_flair_files.shape[0]

    is3D = True

    file_path_prefix = 'train'  # can be "validation"
    if is3D:
        file_path_prefix += '_3D'
    verbose = True

    # Generate tfrecord writer
    result_tf_file = output_tf_records_path + file_path_prefix + '.tfrecord'

    if verbose:
        print("Serializing {:d} examples into {}".format(num_files, result_tf_file))

    # iterate over each sample, and serialize it as ProtoBuf.
    num_shards = int(num_files / SHARD_SIZE)
    print('Total number of shards is ' + str(num_shards))
    print('Expected number of files: ' + str(num_shards * SHARD_SIZE))


    def process_one_shard(shard):
        print("processing shard number %d *****" % shard)
        if is3D:
            X = parallel_load_ukb_3D_multimodal(t1_files[(shard * SHARD_SIZE):((shard + 1) * SHARD_SIZE)],
                                                t2_flair_files[(shard * SHARD_SIZE):((shard + 1) * SHARD_SIZE)])
        else:
            X = parallel_load_ukb_multimodal(t1_files[(shard * SHARD_SIZE):((shard + 1) * SHARD_SIZE)],
                                             t2_flair_files[(shard * SHARD_SIZE):((shard + 1) * SHARD_SIZE)])
            np.random.shuffle(X)  # shuffling input dataset by rows

        print(X.shape)
        output_filename = '%s-%.5d-of-%.5d' % (result_tf_file, shard, num_shards)
        writer = tf.python_io.TFRecordWriter(output_filename)
        for idx in range(X.shape[0]):
            x = X[idx]
            height, width = x.shape[0], x.shape[1]
            depth = None
            if is3D:
                depth = x.shape[2]
            x_reshaped = x.flatten()
            example = _convert_to_example(x_reshaped, height, width, depth)
            serialized = example.SerializeToString()
            writer.write(serialized)
        del X
        if verbose:
            print("Writing {} done!".format(output_filename))


    for shard in range(num_shards - 1):
        process_one_shard(shard)
