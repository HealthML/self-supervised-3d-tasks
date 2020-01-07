import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from keras.applications.vgg19 import VGG19
from keras.layers import Input, Dropout, Dense

from keras import Model
from keras.utils import Sequence, to_categorical
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class KaggleGenerator(Sequence):
    def __init__(self, csvDescriptor=Path("/mnt/mpws2019cl1/kaggle_retina/train/trainLabels.csv"),
                 base_path=Path("/mnt/mpws2019cl1/kaggle_retina/train"), batch_size=1,
                 label_column="level", resize_to=(256, 256), num_classes=5):
        self.dataset = pd.read_csv(csvDescriptor)
        self.batch_size = batch_size
        self.dataset_len = len(self.dataset.index)
        self.n_batches = int(math.ceil(self.dataset_len / batch_size))
        self.label_column = label_column
        self.num_classes = num_classes
        self.base_path = Path(base_path)
        self.resize_width = resize_to[0] if resize_to[0] > 32 else 32
        self.resize_height = resize_to[1] if resize_to[1] > 32 else 32
        self.c = 0

    def __len__(self):
        return self.n_batches

    def load_image(self, index):
        path = self.base_path / self.dataset.iloc[index][0]
        path = path.with_suffix(".jpeg")
        image = Image.open(path)
        image = image.resize((self.resize_width, self.resize_height), resample=Image.LANCZOS)
        return np.array(image)

    def __getitem__(self, index):
        X_t = []
        Y_t = []
        for c in range(index, min(index + self.batch_size, self.dataset_len)):
            X_t.append(self.load_image(c))
            Y_t.append(self.dataset.iloc[c][self.label_column])
        return np.array(X_t), to_categorical(np.array(Y_t), num_classes=self.num_classes)


class F1Metric(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f"— val_f1:{_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}")
        return


def get_cnn_baseline_model(shape=(256, 256, 3,)):
    """
    make a vgg19 keras model
    Args:
       shape: shape of the input data (has to be 2d + channels)

    Returns:
       keras Model() instance, compiled / ready to train
    """
    inputs = Input(shape=shape)
    vgg = VGG19(include_top=False, weights="imagenet")(inputs)
    vgg.trainable = False
    x = vgg
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(5, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    gen = KaggleGenerator(batch_size=8)
    # f1 = F1Metric()
    model = get_cnn_baseline_model()
    model.fit_generator(generator=gen, epochs=501, callbacks=[])
