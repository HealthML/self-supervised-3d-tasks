from datetime import datetime
from pathlib import Path

import numpy as np
import seaborn as sns
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3, VGG16, VGG19
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import f1_score, precision_score, recall_score

from self_supervised_3d_tasks.custom_preprocessing.retina_preprocess import blur_and_subtract, apply_to_x
from self_supervised_3d_tasks.data.kaggle_retina_data import KaggleGenerator
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.keras_algorithms.custom_utils import apply_prediction_model

sns.set()

# img_dim = 384
# csvDescriptor = Path("/mnt/mpws2019cl1/kaggle_retina/train/trainLabels.csv")
# base_path = Path("/mnt/mpws2019cl1/kaggle_retina/train/resized_384")

NGPUS = 1
batch_size=16
test_split =0.9  # MUST be same like split in cnn_baseline_test.py
val_split =0.95
lr = 0.00003  # choose a smaller learning rate

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


def make_vgg(in_shape, output_layer=-1):  # name="resnet"
    # try a larger model?
    vgg = InceptionV3(include_top=False, input_shape=in_shape, weights="imagenet", pooling="max")
    vgg.summary()
    # vgg.trainable = False
    outputs = vgg.layers[output_layer].output
    model = Model(vgg.input, outputs) # name=f"VGG19_{name}"
    # model.trainable = False
    return model


def get_cnn_baseline_model(shape=(384, 384, 3,), multi_gpu=False, lr=1e-3):
    """
    make a vgg19 keras model
    Args:
       shape: shape of the input data (has to be 2d + channels)

    Returns:
       keras Model() instance, compiled / ready to train
    """
    inputs = Input(shape=shape)
    vgg_in_shape = tuple([int(el) for el in inputs.shape[1:]])
    x = make_vgg(in_shape=vgg_in_shape)(inputs)

    model = apply_prediction_model(inputs, x, multi_gpu=multi_gpu)
    model.compile(optimizer=Adam(lr=lr), loss="mse", metrics=["mae"])


def train():
    aquire_free_gpus(NGPUS)

    output = (
        Path(f"~/workspace/cnn_baseline/run_{datetime.now()}/").expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    output = output / f"model.hdf5"

    gen = KaggleGenerator(batch_size=batch_size, sample_classes_uniform=True, shuffle=True,
                          categorical=False,
                          # pre_proc_func_train=apply_to_x, pre_proc_func_val=apply_to_x,
                          discard_part_of_dataset_split=test_split, split=val_split)
    # csvDescriptor=csvDescriptor, base_path=base_path,
    # we have to discard some data BEFORE sampling because of testing

    checkp = ModelCheckpoint(
        str(output.with_name("intermediate_{epoch:04d}_{val_loss:.2f}_" + output.name)),
        monitor="val_mean_absolute_error",
        period=1,
        save_best_only=True,  # reduce amount of data written
        mode="min",
    )
    model = get_cnn_baseline_model(multi_gpu=NGPUS,lr=lr)
    model.fit_generator(
        generator=gen,
        epochs=500,
        callbacks=[checkp],
        validation_data=gen.get_val_data(),
    )
    model.save(output)


if __name__ == "__main__":
    train()
