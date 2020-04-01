import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.data.numpy_2d_loader import Numpy2DLoader
from self_supervised_3d_tasks.data.numpy_3d_loader import DataGeneratorUnlabeled3D


def show_batch(image_batch, reverse_order=False):
    plt.figure(figsize=(10, 10))
    dim = int(np.sqrt(len(image_batch)))

    if dim * dim < len(image_batch):
        dim += 1

        if reverse_order:
            raise ValueError("reverse order only for squared sizes")

    for n in range(len(image_batch)):
        ax = plt.subplot(dim, dim, n + 1)

        if reverse_order:
            row = n // dim
            rest = n % dim

            n = rest * dim + row

        plt.imshow(image_batch[n])
        plt.axis("off")

    plt.show()

def display_slice(image, dim_to_slice, slice_idx, plot_square=False):
    n = len(image)

    for i in range(n):
        img = image[i]
        if plot_square:
            dim = int(np.sqrt(n))
            if dim * dim < n:
                dim += 1

            ax = plt.subplot(dim, dim, i + 1)
        else:
            ax = plt.subplot(n, 1, i + 1)

        idx = [
            slice_idx if dim == dim_to_slice else slice(None)
            for dim in range(img.ndim)
        ]
        im = np.squeeze(img[idx], axis=2)
        ax.axis('off')
        ax.imshow(im, cmap="inferno")

    plt.show()


def plot_3d(image, dim_to_animate, plot_square=False, step=1):
    min_value = [image[i].min() for i in range(len(image))]
    max_value = [image[i].max() for i in range(len(image))]
    print(f"max values: {max_value}, min values: {min_value}")

    plt.figure(figsize=(10, 10))

    n = len(image)
    ax = []
    frame = []
    ani = []

    for i in range(n):
        if plot_square:
            dim = int(np.sqrt(n))
            if dim * dim < n:
                dim += 1

            ax.append(plt.subplot(dim, dim, i + 1))
        else:
            ax.append(plt.subplot(n, 1, i + 1))

        frame.append(None)
        ani.append(-1)

    while True:
        for i in range(n):
            img = image[i]
            ani[i] += step

            if ani[i] >= img.shape[dim_to_animate]:
                ani[i] = -1

            idx = [
                ani[i] if dim == dim_to_animate else slice(None)
                for dim in range(img.ndim)
            ]
            im = np.squeeze(img[idx], axis=2)

            if frame[i] is None:
                init = np.zeros(im.shape)
                init[0, 0] = max_value[i]
                init[0, 1] = min_value[i]
                frame[i] = ax[i].imshow(init, cmap="inferno")
            else:
                frame[i].set_data(im)

        #time.sleep(0.05)
        plt.pause(0.01)
        plt.draw()


def get_data_norm(path):
    img = nib.load(path)
    img = img.get_fdata()
    img = np.expand_dims(img, axis=-1)

    img = (img - img.min()) / (img.max() - img.min())

    return img


def get_data_npy(path):
    img = np.load(path)
    return img


def get_data_norm_npy(path):
    img = np.load(path)
    img = (img - img.min()) / (img.max() - img.min())

    return img

def test_exppp():
    path = "/mnt/mpws2019cl1/Task07_Pancreas/images_resized_128_bbox_labeled/train"

    import self_supervised_3d_tasks.algorithms.exemplar as exp
    instance = exp.create_instance(data_is_3D=True, data_dim=128, sample_neg_examples_from="dataset")
    gen = get_data_generators(path, DataGeneratorUnlabeled3D, train_data_generator_args=
    {
        "pre_proc_func": instance.get_training_preprocessing()[0],
        "shuffle": True,
        "batch_size": 1
    })

    x_batch, y_batch = gen[0]
    plot_3d(x_batch[0], 2, step=3)

def get_2d_loader(f_train):
    # data_dir = "/mnt/mpws2019cl1/kaggle_retina_2019/images/resized_224"
    data_dir = "/mnt/mpws2019cl1/pancreas_data/images_slices_128_labeled/img_single"

    gen = get_data_generators(data_dir,
                        train_data_generator_args={"batch_size": 256,
                                                   "pre_proc_func": f_train},
                        data_generator=Numpy2DLoader)
    return gen

def test_2d_algorithms():
    import self_supervised_3d_tasks.algorithms.rotation as algo

    algorithm_def = algo.create_instance(sample_neg_examples_from="dataset")
    f_train, f_val = algorithm_def.get_training_preprocessing()
    f_id = lambda x,y: (x,y)

    gen = get_2d_loader(f_id)
    batch = gen[0]
    print(batch[0].shape)

    for x in range(15):
        img = batch[0][10+x*2]
        mask = batch[1][10+x * 2]
        print(img.max())
        print(img.min())
        print(mask.shape)
        show_batch([np.squeeze(img, axis=-1), mask])

    # CPC
    # show_batch(batch[0][0][0])
    # show_batch(batch[0][1][0])
    # print(batch[1][0])

if __name__ == "__main__":
    test_2d_algorithms()