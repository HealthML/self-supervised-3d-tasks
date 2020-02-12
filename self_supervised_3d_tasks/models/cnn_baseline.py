from datetime import datetime
from pathlib import Path

import numpy as np
import seaborn as sns
from tensorflow.keras.layers import Dense
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from self_supervised_3d_tasks.data.kaggle_retina_data import get_kaggle_train_generator
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.keras_algorithms.custom_utils import apply_prediction_model
from self_supervised_3d_tasks.keras_models.fully_connected import fully_connected_big

sns.set()

# img_dim = 384
# csvDescriptor = Path("/mnt/mpws2019cl1/kaggle_retina/train/trainLabels.csv")
# base_path = Path("/mnt/mpws2019cl1/kaggle_retina/train/resized_384")

NGPUS = 1
batch_size = 16
val_split = 0.95
lr = 1e-3  # 0.00003  # choose a smaller learning rate


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
    vgg = InceptionV3(include_top=False, input_shape=in_shape, weights=None, pooling="max")
    # vgg.trainable = False
    outputs = vgg.layers[output_layer].output
    model = Model(vgg.input, outputs)  # name=f"VGG19_{name}"
    # model.trainable = False
    model.summary()
    return model


def get_cnn_baseline_model(shape=(384, 384, 3,)):
    """
    make a vgg19 keras model
    Args:
       shape: shape of the input data (has to be 2d + channels)

    Returns:
       keras Model() instance, compiled / ready to train
    """
    vgg = make_vgg(in_shape=shape)

    fc_in = Input(vgg.output_shape[1:])
    fc = fully_connected_big(fc_in)
    fc = Dense(1, activation="relu")(fc)
    pred_model = Model(fc_in, fc)

    # pred_model = apply_prediction_model(input_shape=vgg.output_shape)

    model = Sequential(layers=[vgg, pred_model])
    model.compile(optimizer=Adam(lr=lr), loss="mse", metrics=["mae"])
    pred_model.summary()
    model.summary()
    return model


def train():
    aquire_free_gpus(NGPUS)

    output = (
        Path(f"~/workspace/cnn_baseline/run_{datetime.now()}/").expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    output = output / f"model.hdf5"

    f = lambda x, y: (x, y)
    print("train generator")
    gen = get_kaggle_train_generator(batch_size, val_split, f, f)
    print("done with that")

    checkp = ModelCheckpoint(
        str(output.with_name("intermediate_{epoch:04d}_{val_loss:.2f}_" + output.name)),
        monitor="val_loss",
        period=1,
        save_best_only=True,  # reduce amount of data written
        mode="min",
    )
    model = get_cnn_baseline_model()
    model.fit_generator(
        generator=gen,
        epochs=500,
        callbacks=[checkp],
        validation_data=gen.get_val_data(),
    )
    model.save(output)


if __name__ == "__main__":
    train()
