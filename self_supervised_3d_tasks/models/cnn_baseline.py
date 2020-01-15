import math
from pathlib import Path
from datetime import datetime
import os

import numpy as np
import pandas as pd
from PIL import Image
from functools import lru_cache

from keras.applications.vgg19 import VGG19
from keras.layers import Input, Dropout, Dense, Flatten

from keras import Model
from keras.utils import Sequence, to_categorical, multi_gpu_model
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus

class KaggleGenerator(Sequence):
    def __init__(
            self,
            csvDescriptor=Path("/mnt/mpws2019cl1/kaggle_retina/train/trainLabels.csv"),
            base_path=Path("/mnt/mpws2019cl1/kaggle_retina/train/resized"),
            batch_size=1,
            label_column="level",
            resize_to=(256, 256),
            num_classes=5,
            split=False,
            shuffle=False,
            pre_proc_func_train=None,
            pre_proc_func_val=None
    ):
        self.pre_proc_func_train = pre_proc_func_train
        self.pre_proc_func_val = pre_proc_func_val
        self.dataset = pd.read_csv(csvDescriptor)
        if shuffle:
            self.dataset = self.dataset.sample(frac=1)

        self.batch_size = batch_size
        self.split = split
        self.dataset_len = len(self.dataset.index)
        self.train_len = self.dataset_len
        if self.split:
            splitpoint = math.floor(self.dataset_len * split)
            self.train_len = splitpoint
            self.offset = splitpoint

        self.n_batches = int(math.ceil(self.train_len / batch_size))
        self.label_column = label_column
        self.num_classes = num_classes
        self.base_path = Path(base_path)
        self.resize_width = resize_to[0] if resize_to[0] > 32 else 32
        self.resize_height = resize_to[1] if resize_to[1] > 32 else 32

    def __len__(self):
        return self.n_batches

    def load_image(self, index):
        path = self.base_path / self.dataset.iloc[index][0]
        path = path.with_suffix(".jpeg")
        image = Image.open(path)
        if image.width != self.resize_width or image.height != self.resize_height:
            image = image.resize(
                (self.resize_width, self.resize_height), resample=Image.LANCZOS
            )
        return np.array(image)

    def get_val_data(self, debug=False):
        assert (
            self.split
        ), "To use Validation Data a fractional split has to be given initially."
        endpoint = self.dataset_len if not debug else self.offset + 200
        X_t = []
        Y_t = []
        for c in range(self.offset, endpoint):  # todo: remove val set binding
            X_t.append(self.load_image(c))
            Y_t.append(self.dataset.iloc[c][self.label_column])

        data_x = np.array(X_t)
        data_y = to_categorical(np.array(Y_t), num_classes=self.num_classes)

        if self.pre_proc_func_val:
            data_x, data_y = self.pre_proc_func_val(data_x, data_y)

        return (
            data_x,
            data_y,
        )

    def __getitem__(self, index):
        X_t = []
        Y_t = []
        for c in range(index, min(index + self.batch_size, self.train_len)):
            X_t.append(self.load_image(c))
            Y_t.append(self.dataset.iloc[c][self.label_column])

        data_x = np.array(X_t)
        data_y = to_categorical(np.array(Y_t), num_classes=self.num_classes)

        if self.pre_proc_func_train:
            data_x, data_y = self.pre_proc_func_train(data_x, data_y)

        return (
            data_x,
            data_y,
        )


class F1Metric(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (
            np.asarray(self.model.predict(self.model.validation_data[0]))
        ).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(
            f"— val_f1:{_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}"
        )
        return


def make_vgg(in_shape, name, output_layer=-1):
    vgg = VGG19(include_top=False, input_shape=in_shape, weights="imagenet")
    # vgg.trainable = False
    outputs = vgg.layers[output_layer].output
    model = Model(vgg.input, outputs, name=f"VGG19_{name}")
    # model.trainable = False
    return model


def get_cnn_baseline_model(shape=(256, 256, 3,), multi_gpu=False):
    """
    make a vgg19 keras model
    Args:
       shape: shape of the input data (has to be 2d + channels)

    Returns:
       keras Model() instance, compiled / ready to train
    """
    inputs = Input(shape=shape)
    vgg_in_shape = tuple([int(el) for el in inputs.shape[1:]])
    x = make_vgg(in_shape=vgg_in_shape, name="Baseline_Freezed")(inputs)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(5, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    if multi_gpu:
        model = multi_gpu_model(model, gpus=NGPUS)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model


if __name__ == "__main__":
    NGPUS = 3
    aquire_free_gpus(NGPUS)

    output = (
        Path(f"~/workspace/cnn_baseline/run_{datetime.now()}/").expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    output = output / f"model.hdf5"
    gen = KaggleGenerator(batch_size=64, split=0.66, shuffle=False)
    checkp = ModelCheckpoint(
        str(output.with_name("intermediate_{epoch:04d}_{acc:.2f}_" + output.name)),
        monitor="accuracy",
        period=1,
        mode="max",
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=0.001
    )
    model = get_cnn_baseline_model(multi_gpu=True)
    model.fit_generator(
        generator=gen,
        epochs=500,
        callbacks=[checkp, reduce_lr],
        validation_data=gen.get_val_data(),
    )
    model.save(output)
