import csv
import os

import tensorflow.keras as keras

from pathlib import Path
import matplotlib.pyplot as plt
import pandas
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, concatenate

from datetime import datetime

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus

from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc

from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import preprocess, resize


def run_single_test(train_split, test_split, load_weights, freeze_weights, save_ckpt=False):
    epochs = 3
    code_size = 128
    lr = 1e-3
    terms = 3
    predict_terms = 3
    image_size = 46
    batch_size = 64
    n_channels = 3
    crop_size = 186
    split_per_side = 7
    img_shape = (image_size, image_size, n_channels)

    f_train = lambda x, y: (preprocess(resize(x, 192), crop_size, split_per_side, is_training=False), y)
    f_val = lambda x, y: (preprocess(resize(x, 192), crop_size, split_per_side, is_training=False), y)

    gen = KaggleGenerator(batch_size=batch_size, split=train_split, shuffle=False, pre_proc_func_train=f_train,
                          pre_proc_func_val=f_val)
    gen_test = KaggleGenerator(batch_size=batch_size, split=1.0-test_split, shuffle=False, pre_proc_func_train=f_train,
                               pre_proc_func_val=f_val)
    X_test, Y_test = gen_test.get_val_data()

    cpc_model, enc_model = network_cpc(image_shape=img_shape, terms=terms,
                                       predict_terms=predict_terms, code_size=code_size, learning_rate=lr)

    if load_weights:
        cpc_model.load_weights('/home/Julius.Severin/workspace/self-supervised-transfer-learning/cpc_retina/'
                          'weights-improvement-1000-0.48.hdf5')

    if freeze_weights:
        enc_model.trainable = False

    layer_in = keras.layers.Input((split_per_side * split_per_side,) + img_shape)
    layer_out = keras.layers.TimeDistributed(enc_model)(layer_in)

    x = Flatten()(layer_out)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(5, activation="sigmoid")(x)

    model = Model(inputs=layer_in, outputs=x)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if save_ckpt:
        output = (
            Path(f"~/workspace/cpc_test/run_{datetime.now()}/").expanduser().resolve()
        )
        output.mkdir(parents=True, exist_ok=True)
        output = output / f"model.hdf5"

        checkp = ModelCheckpoint(
            str(output.with_name("intermediate_{epoch:04d}_{acc:.2f}_" + output.name)),
            monitor="accuracy",
            period=1,
            mode="max",
        )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=0.001
    )

    model.fit_generator(
        generator=gen,
        epochs=epochs,
        callbacks=([checkp, reduce_lr] if save_ckpt else [reduce_lr])
    )

    if save_ckpt:
        model.save(str(output))

    scores = model.evaluate(X_test, Y_test)
    print("model accuracy: {}".format(scores[1]))

    return scores[1]


def write_result(row):
    with open('results2.csv', 'a') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',')
        result_writer.writerow(row)


def draw_curve():
    df = pandas.read_csv('results2.csv')

    #print(df["Train Split"].isin([0,0.1,0.2,0.3,0.4]))
    #df = df[df["Train Split"].isin([0,0.1,0.2,0.3,0.4])]

    plt.plot(df["Train Split"], df["Weights initialized"], label='CPC pretrained')
    plt.plot(df["Train Split"], df["Weights random"], label='Random')

    plt.legend()
    plt.show()

    print(df["Train Split"])


def run_complex_test():
    aquire_free_gpus(2)
    test_split = 0.2
    results = []
    repetitions = 5

    write_result(["Train Split", "Weights initialized", "Weights random"])

    for train_split in [0.5,1,2,5,10,25,50,80]:
        percentage = 0.01 * train_split
        print("running test for: {}%".format(train_split))

        a_s = []
        b_s = []

        for i in range(repetitions):
            a = run_single_test(percentage, test_split, True, False)
            b = run_single_test(percentage, test_split, False, False)

            print("train split:{} model accuracy initialized: {} random: {}".format(percentage, a, b))
            a_s.append(a)
            b_s.append(b)

        data = [str(train_split)+"%", np.mean(np.array(a_s)), np.mean(np.array(b_s))]
        results.append(data)
        write_result(data)


if __name__ == "__main__":
    draw_curve()
    print("done")
