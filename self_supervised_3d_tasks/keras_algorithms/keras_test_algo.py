import csv
import gc
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras import Sequential
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras import backend as K

from self_supervised_3d_tasks.data.kaggle_retina_data import get_kaggle_test_generator, \
    get_kaggle_train_generator
from self_supervised_3d_tasks.keras_algorithms.custom_utils import init, apply_prediction_model
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import keras_algorithm_list


def score(y, y_pred):
    return "kappa score", cohen_kappa_score(y, y_pred, labels=[0, 1, 2, 3, 4], weights="quadratic")


def get_dataset_train(dataset_name, batch_size, f_train, f_val, train_split):
    if dataset_name == "kaggle_retina":
        gen_train = get_kaggle_train_generator(batch_size, train_split, f_train, f_val)
    else:
        raise ValueError("not implemented")

    return gen_train


def get_dataset_test(dataset_name, batch_size, f_train, f_val):
    if dataset_name == "kaggle_retina":
        gen_test = get_kaggle_test_generator(batch_size, f_train, f_val)
        x_test, y_test = gen_test.get_val_data()
    else:
        raise ValueError("not implemented")

    return x_test, y_test


def run_single_test(algorithm_def, dataset_name, train_split, load_weights, freeze_weights, x_test, y_test, lr,
                    batch_size, epochs, epochs_warmup, model_checkpoint):
    f_train, f_val = algorithm_def.get_finetuning_preprocessing()
    gen = get_dataset_train(dataset_name, batch_size, f_train, f_val, train_split)

    if load_weights:
        enc_model, cleanup_models = algorithm_def.get_finetuning_model(model_checkpoint)
    else:
        enc_model, cleanup_models = algorithm_def.get_finetuning_model()

    pred_model = apply_prediction_model(input_shape=enc_model.output_shape, lr=lr)
    model = Sequential(layers=[enc_model, pred_model])

    if freeze_weights or load_weights:
        enc_model.trainable = False

    if load_weights:
        assert epochs_warmup > epochs, "warmup epochs must be smaller than epochs"

        model.fit_generator(generator=gen, epochs=epochs_warmup)
        epochs = epochs - epochs_warmup

        enc_model.trainable = True

    model.fit_generator(generator=gen, epochs=epochs)

    y_pred = model.predict(x_test)
    y_pred = np.rint(y_pred)
    s_name, result = score(y_test, y_pred)

    # cleanup
    del model
    for i in sorted(range(len(cleanup_models)), reverse=True):
        del cleanup_models[i]

    K.clear_session()

    for i in range(3):
        gc.collect()

    print("{} score: {}".format(s_name, result))
    return result


def write_result(base_path, row):
    with open(base_path + '_results.csv', 'a') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',')
        result_writer.writerow(row)


def draw_curve(name):
    # TODO: load multiple algorithms here
    # helper function to plot results curve
    df = pandas.read_csv(name + '_results.csv')

    plt.plot(df["Train Split"], df["Weights initialized"], label=name + ' pretrained')
    plt.plot(df["Train Split"], df["Weights random"], label='Random')
    plt.plot(df["Train Split"], df["Weights freezed"], label=name + 'Freezed')

    plt.legend()
    plt.show()

    print(df["Train Split"])


def run_complex_test(algorithm, dataset_name, root_config_file, model_checkpoint, epochs=5, repetitions=2, batch_size=2,
                     exp_splits=(100, 50, 25, 12.5, 6.25), lr=0.00003, epochs_warmup=2, **kwargs):
    kwargs["model_checkpoint"] = model_checkpoint
    kwargs["root_config_file"] = root_config_file

    shutil.copy2(root_config_file, model_checkpoint+"test_config.json")
    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    results = []

    write_result(model_checkpoint, ["Train Split", "Weights freezed", "Weights initialized", "Weights random"])
    f_train, f_val = algorithm_def.get_finetuning_preprocessing()
    x_test, y_test = get_dataset_test(dataset_name, batch_size, f_train, f_val)

    for train_split in exp_splits:
        percentage = 0.01 * train_split
        print("running test for: {}%".format(train_split))

        a_s = []
        b_s = []
        c_s = []

        for i in range(repetitions):
            # load and freeze weights
            a = run_single_test(algorithm_def, dataset_name, percentage, True, True, x_test, y_test, lr,
                                batch_size, epochs, epochs_warmup, model_checkpoint)

            # load weights and train
            b = run_single_test(algorithm_def, dataset_name, percentage, True, False, x_test, y_test, lr,
                                batch_size, epochs, epochs_warmup, model_checkpoint)

            # random initialization
            c = run_single_test(algorithm_def, dataset_name, percentage, False, False, x_test, y_test, lr,
                                batch_size, epochs, epochs_warmup, model_checkpoint)

            print("train split:{} model accuracy freezed: {}, initialized: {}, random: {}".format(percentage, a, b, c))

            a_s.append(a)
            b_s.append(b)
            c_s.append(c)

        data = [str(train_split) + "%", np.mean(np.array(a_s)), np.mean(np.array(b_s)), np.mean(np.array(c_s))]
        results.append(data)
        write_result(model_checkpoint, data)


if __name__ == "__main__":
    # draw_curve("jigsaw")
    init(run_complex_test, "test")
