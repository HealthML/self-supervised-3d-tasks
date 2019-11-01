import functools

import absl.flags as flags
import datasets
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from algorithms.patch_model_preprocess import get_crop_patches_fn


def plot_sequences(x, y, labels=None, output_path=None):

    ''' Draws a plot where sequences of numbers can be studied conveniently '''

    images = np.concatenate([x, y], axis=1)
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(images[n_b, n_t, :, :, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b],fontdict={'color':'white'})

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()

def get_lena():
    img = img = tf.io.read_file('data_util/resources/lena.jpg')
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return {"image": tf.image.resize(img, [256, 256])}

def test_mnist_data_generator():
    flags.DEFINE_string('dataset', 'cpc_test', 'Which dataset to use, typically '
                                               '`imagenet`.')

    flags.DEFINE_string('dataset_dir', 'data_util/tf_records', 'Location of the dataset files.')
    flags.DEFINE_string('preprocessing', None, "")
    flags.DEFINE_integer('random_seed', 1, "")

    f = functools.partial(
        datasets.get_data,
        split_name='train',
        is_training=True,
        num_epochs=1,
        shuffle=False,
        drop_remainder=True)

    result = f({'batch_size': 8})
    print(result)

    iterator = result.make_one_shot_iterator()
    el = iterator.get_next()
    with tf.Session() as sess:
        batch = sess.run(el)
        print(batch["example"]["image/encoded"].shape)

        plot_sequences(batch["example"]["image/encoded"], batch["example"]["image/encoded_pred"],
                       batch["example"]["image/labels"], output_path=r'testXXX.png')


def show_batch(image_batch):
    plt.figure(figsize=(10,10))
    dim = int(np.sqrt(len(image_batch)))

    if dim * dim < len(image_batch):
        dim += 1

    for n in range(len(image_batch)):
        ax = plt.subplot(dim, dim, n + 1)
        plt.imshow(image_batch[n])
        plt.axis('off')

    plt.show()

def show_img(img):
    print(img.shape)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

def test_preprocessing():
    with tf.Session() as sess:
        f = get_crop_patches_fn(is_training=True, split_per_side=7, patch_jitter=-32)
        patches = sess.run(f(get_lena()))
        print(patches["image"].shape)
        show_batch(patches["image"])

test_preprocessing()