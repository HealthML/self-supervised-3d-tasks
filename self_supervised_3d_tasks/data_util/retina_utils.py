import math
import os
from os.path import expanduser

import tensorflow as tf
import numpy as np
from PIL import Image

SHARD_SIZE = 1024

resize = True
resize_w = 128
resize_l = 128

import matplotlib.pyplot as plt
from skimage.transform import resize


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


def _string_feature(value):
    """Wrapper for inserting float features into Example proto."""
    encbytes = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[encbytes]))


def _convert_to_example(image_buffer, fname, height, width, channels):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/data': _float_feature(image_buffer),
        'image/filename': _string_feature(fname)}))
    return example


def process_one_shard(n_shards, current_shard_id, current_shard, result_tf_file):
    print("processing shard number %d *****" % current_shard_id)
    output_filename = '%s-%.5d-of-%.5d' % (result_tf_file, current_shard_id, n_shards)
    writer = tf.python_io.TFRecordWriter(output_filename)

    for i in range(len(current_shard)):
        img = current_shard[i][0]
        filename = current_shard[i][1]

        example = _convert_to_example(img.flatten(), filename, *img.shape)

        serialized = example.SerializeToString()
        writer.write(serialized)
    print("Writing {} done!".format(output_filename))


if __name__ == "__main__":
    # generate TF Records, we have train / valid / test
    output_tf_records_path = "/mnt/mpws2019cl1/retinal_fundus/retina_tf_records/max_256/"
    directory = expanduser("~/retinal_fundus/left/max_256/")

    corrupted_images = []
    n_images = len(os.listdir(directory))
    print("n images: "+str(n_images))
    n_shards = math.ceil(float(n_images) / float(SHARD_SIZE))
    print("n shards: "+str(n_shards))

    if not os.path.exists(output_tf_records_path):
        os.mkdir(output_tf_records_path)

    result_tf_file = output_tf_records_path + '.tfrecord'

    current_shard = []
    current_shard_id = 1

    for filename in os.listdir(directory):
        try:
            im_frame = Image.open("{}/{}".format(directory, filename))
            print("loading: " + filename)

            im_frame.load()
        except:
            corrupted_images.append(filename)
            print("LOADING FAILED FOR: {}".format(filename))
            continue

        img = np.asarray(im_frame, dtype="float32")
        img /= 255

        if resize:
            img = resize(img, (resize_w, resize_l))

        current_shard.append((img, filename))

        if len(current_shard) == SHARD_SIZE:
            process_one_shard(n_shards, current_shard_id, current_shard, result_tf_file)
            current_shard.clear()
            current_shard_id += 1

    if len(current_shard) > 0:
        process_one_shard(n_shards, current_shard_id, current_shard, result_tf_file)

    print("could not load following {} images: {}".format(len(corrupted_images), corrupted_images))
    print("final count: {}".format(n_images-len(corrupted_images)))
    print("Done")