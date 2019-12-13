import glob
import multiprocessing

import nibabel as nib
import numpy as np
import skimage.transform as skTrans
import tensorflow as tf
from joblib import Parallel, delayed

SHARD_SIZE = 1024


def parallel_load_brats_no_labels(path, multimodal=True):
    """
    loader function for self-supervision tasks
    :param path:
    :return:
    """
    t1ce_files = sorted(glob.glob(path + "*_t1ce.nii.gz", recursive=True))
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    print("loading " + str(len(flair_files)) + " mri scans.")

    num_cores = multiprocessing.cpu_count()
    all_slices = list()
    if multimodal:
        t1_files = sorted(glob.glob(path + "*_t1.nii.gz", recursive=True))
        t2_files = sorted(glob.glob(path + "*_t2.nii.gz", recursive=True))
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_multimodal)(flair_files, i, t1_files, t1ce_files, t2_files) for i in
            range(len(t2_files)))

        for mm_scan in results:
            if mm_scan[0].shape[2] == mm_scan[1].shape[2] == mm_scan[2].shape[2] == mm_scan[3].shape[2]:
                for z in range(mm_scan[0].shape[2]):
                    t1ce_image = mm_scan[0][:, :, z]
                    flair_image = mm_scan[1][:, :, z]
                    t1_image = mm_scan[2][:, :, z]
                    t2_image = mm_scan[3][:, :, z]
                    stacked_array = np.stack([t1ce_image, flair_image, t1_image, t2_image], axis=-1)
                    all_slices.append(stacked_array)
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_two_modal)(flair_files, i, t1ce_files)
            for i in range(len(flair_files)))

        for mm_scan in results:
            if mm_scan[0].shape[2] == mm_scan[1].shape[2]:
                for z in range(mm_scan[0].shape[2]):
                    t1ce_image = mm_scan[0][:, :, z]
                    flair_image = mm_scan[1][:, :, z]
                    stacked_array = np.stack([t1ce_image, flair_image], axis=-1)
                    all_slices.append(stacked_array)

    return np.array(all_slices)


def parallel_load_brats_3D_no_labels(path, multimodal=True):
    """
    loader function for self-supervision tasks
    :param path:
    :return:
    """
    t1ce_files = sorted(glob.glob(path + "*_t1ce.nii.gz", recursive=True))
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    print("loading " + str(len(flair_files)) + " mri scans.")

    num_cores = multiprocessing.cpu_count()
    all_scans = list()
    if multimodal:
        t1_files = sorted(glob.glob(path + "*_t1.nii.gz", recursive=True))
        t2_files = sorted(glob.glob(path + "*_t2.nii.gz", recursive=True))
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_multimodal)(flair_files, i, t1_files, t1ce_files, t2_files) for i in
            range(len(t2_files)))

        for mm_scan in results:
            t1ce_image = mm_scan[0]
            flair_image = mm_scan[1]
            t1_image = mm_scan[2]
            t2_image = mm_scan[3]
            stacked_array = np.stack([t1ce_image, flair_image, t1_image, t2_image], axis=-1)
            all_scans.append(stacked_array)
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_two_modal)(flair_files, i, t1ce_files)
            for i in range(len(flair_files)))

        for mm_scan in results:
            t1ce_image = mm_scan[0]
            flair_image = mm_scan[1]
            stacked_array = np.stack([t1ce_image, flair_image], axis=-1)
            all_scans.append(stacked_array)

    return np.array(all_scans)


def parallel_load_brats_with_labels(path, multimodal=False):
    """
    loader function for segmentation tasks
    :param path:
    :param multimodal:
    :return:
    """
    t1ce_files = sorted(glob.glob(path + "*_t1ce.nii.gz", recursive=True))
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    seg_files = sorted(glob.glob(path + "*_seg.nii.gz", recursive=True))
    print("loading " + str(len(seg_files)) + " mri scans.")

    num_cores = multiprocessing.cpu_count()
    all_slices = list()
    all_masks = list()
    if multimodal:
        t1_files = sorted(glob.glob(path + "*_t1.nii.gz", recursive=True))
        t2_files = sorted(glob.glob(path + "*_t2.nii.gz", recursive=True))
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_multimodal_with_mask)(flair_files, i, t1_files, t1ce_files, t2_files, seg_files)
            for i in range(len(flair_files)))
        for mm_scan in results:
            if mm_scan[0].shape[2] == mm_scan[1].shape[2] == mm_scan[2].shape[2] == mm_scan[3].shape[2] == \
                    mm_scan[4].shape[2]:
                for z in range(mm_scan[0].shape[2]):
                    t1ce_image = mm_scan[0][:, :, z]
                    flair_image = mm_scan[1][:, :, z]
                    t1_image = mm_scan[2][:, :, z]
                    t2_image = mm_scan[3][:, :, z]
                    seg_image = mm_scan[4][:, :, z]
                    stacked_array = np.stack([t1ce_image, flair_image, t1_image, t2_image], axis=-1)
                    all_slices.append(stacked_array)
                    all_masks.append(seg_image)
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_two_modal_with_mask)(flair_files, i, t1ce_files, seg_files)
            for i in range(len(flair_files)))
        for mm_scan in results:
            if mm_scan[0].shape[2] == mm_scan[1].shape[2] == mm_scan[2].shape[2]:
                for z in range(mm_scan[0].shape[2]):
                    t1ce_image = mm_scan[0][:, :, z]
                    flair_image = mm_scan[1][:, :, z]
                    seg_image = mm_scan[2][:, :, z]
                    stacked_array = np.stack([t1ce_image, flair_image], axis=-1)
                    all_slices.append(stacked_array)
                    all_masks.append(seg_image)

    return np.array(all_slices), np.array(all_masks)


def parallel_load_brats_3D_with_labels(path, multimodal=False):
    """
    loader function for segmentation tasks
    :param path:
    :param multimodal:
    :return:
    """
    t1ce_files = sorted(glob.glob(path + "*_t1ce.nii.gz", recursive=True))
    flair_files = sorted(glob.glob(path + "*_flair.nii.gz", recursive=True))
    seg_files = sorted(glob.glob(path + "*_seg.nii.gz", recursive=True))
    print("loading " + str(len(seg_files)) + " mri scans.")

    num_cores = multiprocessing.cpu_count()
    all_scans = list()
    all_masks = list()
    if multimodal:
        t1_files = sorted(glob.glob(path + "*_t1.nii.gz", recursive=True))
        t2_files = sorted(glob.glob(path + "*_t2.nii.gz", recursive=True))
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_multimodal_with_mask)(flair_files, i, t1_files, t1ce_files, t2_files, seg_files)
            for i in range(len(flair_files)))
        for mm_scan in results:
            t1ce_image = mm_scan[0]
            flair_image = mm_scan[1]
            t1_image = mm_scan[2]
            t2_image = mm_scan[3]
            seg_image = mm_scan[4]
            stacked_array = np.stack([t1ce_image, flair_image, t1_image, t2_image], axis=-1)
            all_scans.append(stacked_array)
            all_masks.append(seg_image)
    else:
        results = Parallel(n_jobs=num_cores)(
            delayed(read_brats_scan_two_modal_with_mask)(flair_files, i, t1ce_files, seg_files)
            for i in range(len(flair_files)))
        for mm_scan in results:
            t1ce_image = mm_scan[0]
            flair_image = mm_scan[1]
            stacked_array = np.stack([t1ce_image, flair_image], axis=-1)
            seg_image = mm_scan[2]
            all_scans.append(stacked_array)
            all_masks.append(seg_image)

    return np.array(all_scans), np.array(all_masks)


new_resolution = (128, 128, 128)


def read_brats_scan_multimodal(flair_files, i, t1_files, t1ce_files, t2_files):
    t1ce_image, nbbox = read_scan_find_bbox(nib.load(t1ce_files[i]))
    t1ce_image = skTrans.resize(t1ce_image, new_resolution, order=1, preserve_range=True)
    flair_image = read_scan(nbbox, nib.load(flair_files[i]))
    flair_image = skTrans.resize(flair_image, new_resolution, order=1, preserve_range=True)
    t1_image = read_scan(nbbox, nib.load(t1_files[i]))
    t1_image = skTrans.resize(t1_image, new_resolution, order=1, preserve_range=True)
    t2_image = read_scan(nbbox, nib.load(t2_files[i]))
    t2_image = skTrans.resize(t2_image, new_resolution, order=1, preserve_range=True)
    return t1ce_image, flair_image, t1_image, t2_image


def read_brats_scan_two_modal(flair_files, i, t1ce_files):
    t1ce_scan, nbbox = read_scan_find_bbox(nib.load(t1ce_files[i]))
    t1ce_scan = skTrans.resize(t1ce_scan, new_resolution, order=1, preserve_range=True)
    flair_scan = read_scan(nbbox, nib.load(flair_files[i]))
    flair_scan = skTrans.resize(flair_scan, new_resolution, order=1, preserve_range=True)
    return t1ce_scan, flair_scan


def read_brats_scan_multimodal_with_mask(flair_files, i, t1_files, t1ce_files, t2_files, seg_files):
    t1ce_image, nbbox = read_scan_find_bbox(nib.load(t1ce_files[i]))
    t1ce_image = skTrans.resize(t1ce_image, new_resolution, order=1, preserve_range=True)
    flair_image = read_scan(nbbox, nib.load(flair_files[i]))
    flair_image = skTrans.resize(flair_image, new_resolution, order=1, preserve_range=True)
    t1_image = read_scan(nbbox, nib.load(t1_files[i]))
    t1_image = skTrans.resize(t1_image, new_resolution, order=1, preserve_range=True)
    t2_image = read_scan(nbbox, nib.load(t2_files[i]))
    t2_image = skTrans.resize(t2_image, new_resolution, order=1, preserve_range=True)
    seg_image = read_scan(nbbox, nib.load(seg_files[i]), normalize=False)
    seg_image = skTrans.resize(seg_image, new_resolution, order=0, preserve_range=True)
    seg_image = np.asarray(seg_image, dtype=np.int32)
    seg_image[seg_image == 4] = 3
    return t1ce_image, flair_image, t1_image, t2_image, seg_image


def read_brats_scan_two_modal_with_mask(flair_files, i, t1ce_files, seg_files):
    t1ce_image, nbbox = read_scan_find_bbox(nib.load(t1ce_files[i]))
    t1ce_image = skTrans.resize(t1ce_image, new_resolution, order=1, preserve_range=True)
    flair_image = read_scan(nbbox, nib.load(flair_files[i]))
    flair_image = skTrans.resize(flair_image, new_resolution, order=1, preserve_range=True)
    seg_image = read_scan(nbbox, nib.load(seg_files[i]), normalize=False)
    seg_image = skTrans.resize(seg_image, new_resolution, order=0, preserve_range=True)
    seg_image = np.asarray(seg_image, dtype=np.int32)
    seg_image[seg_image == 4] = 3
    return t1ce_image, flair_image, seg_image


def read_scan_find_bbox(nif_file, normalize=True):
    image = nif_file.get_fdata()
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
    if normalize:
        image = norm(image)
    nbbox = np.array([st_x, en_x, st_y, en_y, st_z, en_z]).astype(int)
    return image, nbbox


def read_scan(sbbox, nif_file, normalize=True):
    if normalize:
        image = norm(nif_file.get_fdata()[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]])
    else:
        image = nif_file.get_fdata()[sbbox[0]:sbbox[1], sbbox[2]:sbbox[3], sbbox[4]:sbbox[5]]
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


def _int64_array_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example_no_labels(image_buffer, height, width, multimodal=True):
    """Build an Example proto for an example. """
    if multimodal:
        channels = 4
    else:
        channels = 2

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/encoded': _float_feature(image_buffer)}))
    return example


def _convert_to_example_with_labels(image_buffer, mask_buffer, height, width, multimodal=False):
    """Build an Example proto for an example. """
    if multimodal:
        channels = 4
    else:
        channels = 2

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/mask': _int64_array_feature(mask_buffer),
        'image/encoded': _float_feature(image_buffer)}))
    return example


def np_to_tfrecords_no_labels(X, file_path_prefix, verbose=True, multimodal=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecord'

    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

    np.random.shuffle(X)  # shuffling input dataset by rows

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    shard = 0
    shard_size = SHARD_SIZE
    if X.shape[0] < SHARD_SIZE:
        shard_size = 30
        num_shards = int(X.shape[0] / shard_size)
    else:
        num_shards = int(X.shape[0] / shard_size)
    print('Total number of shards is ' + str(num_shards))
    for idx in range(X.shape[0]):
        if idx % shard_size == 0:
            output_filename = '%s-%.5d-of-%.5d' % (result_tf_file, shard, num_shards)
            writer = tf.python_io.TFRecordWriter(output_filename)
            shard += 1
            print("Created shard {:d}. Working on shard number {:d} now".format(shard - 1, shard))
            print(str(idx) + "/" + str(X.shape[0]))
        x = X[idx]
        height, width = x.shape[0], x.shape[1]
        x_reshaped = x.flatten()
        example = _convert_to_example_no_labels(x_reshaped, height, width, multimodal)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))


def np_to_tfrecords_with_labels(X, Y, file_path_prefix, verbose=True, multimodal=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecord'

    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

    # shuffling input dataset by rows
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    # iterate over each sample, and serialize it with its corresponding mask as ProtoBuf.
    shard = 0
    shard_size = SHARD_SIZE
    if X.shape[0] < SHARD_SIZE:
        shard_size = 30
        num_shards = int(X.shape[0] / shard_size)
    else:
        num_shards = int(X.shape[0] / shard_size)
    print('Total number of shards is ' + str(num_shards))
    for idx in range(X.shape[0]):
        if idx % shard_size == 0:
            output_filename = '%s-%.5d-of-%.5d' % (result_tf_file, shard, num_shards)
            writer = tf.python_io.TFRecordWriter(output_filename)
            shard += 1
            print("Created shard {:d}. Working on shard number {:d} now".format(shard - 1, shard))
            print(str(idx) + "/" + str(X.shape[0]))
        x = X[idx]
        y = Y[idx]
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == y.shape[1]
        height, width = x.shape[0], x.shape[1]
        x_reshaped = x.flatten()
        y_reshaped = y.flatten()
        example = _convert_to_example_with_labels(x_reshaped, y_reshaped, height, width, multimodal)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))

if __name__ == "__main__":
    #################################
    ##      Test and Use Cases     ##
    #################################
    brats_path = "/mnt/30T/brats/MICCAI_BraTS_2018_Data_Training/**/"
    file_prefix = 'train_3D_with_labels'
    with_masks = True

    # brats_path = "/mnt/30T/brats/MICCAI_BraTS_2018_Data_Validation/**/"
    # file_prefix = 'validation_3D_two_modal'
    # with_masks = False

    is3D = True
    multimodal = False  # True: use 4 modalities, False: use only 2 (t1ce + t2flair)
    if with_masks:
        if is3D:
            brats_X, brats_Y = parallel_load_brats_3D_with_labels(brats_path, multimodal)
        else:
            brats_X, brats_Y = parallel_load_brats_with_labels(brats_path, multimodal)
        print(brats_X.shape, brats_Y.shape)

        np_to_tfrecords_with_labels(brats_X, brats_Y, file_prefix, verbose=True, multimodal=multimodal)
    else:
        if is3D:
            brats_data = parallel_load_brats_3D_no_labels(brats_path, multimodal)
        else:
            brats_data = parallel_load_brats_no_labels(brats_path, multimodal)
        print(brats_data.shape)

        np_to_tfrecords_no_labels(brats_data, file_prefix, verbose=True, multimodal=multimodal)
