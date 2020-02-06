import keras
import nibabel as nib
import numpy as np
import skimage.transform as skTrans


def norm(im):
    im = im.astype(np.float32)
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im


def read_scan_find_bbox(image, normalize=True):
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


class DataGeneratorUnlabeled3D(keras.utils.Sequence):

    def __init__(self,
                 data_path,
                 file_list,
                 batch_size=32,
                 shuffle=True,
                 pre_proc_func=None,
                 dim=None):
        self.batch_size = batch_size
        self.list_IDs = file_list
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.path_to_data = data_path
        self.pre_proc_func = pre_proc_func
        self.on_epoch_end()
        self.dim = dim

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(list_IDs_temp)

        return (X, Y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_temp):
        data_x = []
        data_y = None

        for i, file_name in enumerate(list_files_temp):
            path_to_image = "{}/{}".format(self.path_to_data, file_name)
            try:
                img = nib.load(path_to_image)
                img = img.get_fdata()

                if self.dim is not None:
                    img, _ = read_scan_find_bbox(img)
                    img = skTrans.resize(img, self.dim, order=1, preserve_range=True)

                data_x.append(np.expand_dims(img, axis=3)) # we have n_channels = 1
            except Exception as e:
                print("Error while loading image {}.".format(path_to_image))
                print(e)
                continue

        data_x = np.stack(data_x)
        if self.pre_proc_func:
            data_x, data_y = self.pre_proc_func(data_x, data_y)

        return data_x, np.array(data_y)