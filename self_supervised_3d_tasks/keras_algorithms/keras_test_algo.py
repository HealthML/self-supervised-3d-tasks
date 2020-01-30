import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.metrics import cohen_kappa_score

from self_supervised_3d_tasks.data.kaggle_retina_data import KaggleGenerator
from self_supervised_3d_tasks.keras_algorithms.custom_utils import init, apply_prediction_model
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import keras_algorithm_list

epochs = 10
repetitions = 3
batch_size = 16
exp_splits = [0.5, 1, 2, 5, 10, 25, 50, 80, 90]  # different train splits to try in %

test_split = 0.9
val_split = 0.95
NGPUS = 1
lr = 0.00003


def score(y, y_pred):
    return "kappa score", cohen_kappa_score(y, y_pred, labels=[0, 1, 2, 3, 4], weights="quadratic")


def get_dataset(dataset_name, batch_size, f_train, f_val, train_split):
    if train_split > test_split:
        raise ValueError("training data includes testing data")

    if dataset_name == "kaggle_retina":
        gen_train = KaggleGenerator(batch_size=batch_size, sample_classes_uniform=True, shuffle=True,
                                    categorical=False, discard_part_of_dataset_split=train_split, split=val_split,
                                    pre_proc_func_train=f_train, pre_proc_func_val=f_val)
        gen_test = KaggleGenerator(batch_size=batch_size, split=test_split, shuffle=False, categorical=False,
                                   pre_proc_func_train=f_train, pre_proc_func_val=f_val)
        x_test, y_test = gen_test.get_val_data()
    else:
        raise ValueError("not implemented")  # we can only test with kaggle retina atm.

    return gen_train, x_test, y_test


def run_single_test(algorithm, dataset_name, train_split, load_weights, freeze_weights):
    algorithm_def = keras_algorithm_list[algorithm]
    f_train, f_val = algorithm_def.get_finetuning_preprocessing()
    gen, x_test, y_test = get_dataset(dataset_name, batch_size, f_train, f_val, train_split)

    layer_in, x = algorithm_def.get_finetuning_layers(load_weights, freeze_weights)
    model = apply_prediction_model(layer_in, x, multi_gpu=NGPUS, lr=lr)
    model.fit_generator(generator=gen, epochs=epochs)

    y_pred = model.predict(x_test)
    y_pred = np.rint(y_pred)
    s_name, result = score(y_test, y_pred)

    print("{} score: {}".format(s_name, result))
    return result


def write_result(name, row):
    with open(name + '_results.csv', 'a') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',')
        result_writer.writerow(row)


def draw_curve(name):
    # TODO: load multiple algorithms here
    # helper function to plot results curve
    df = pandas.read_csv(name + '_results.csv')

    plt.plot(df["Train Split"], df["Weights initialized"], label='CPC pretrained')
    plt.plot(df["Train Split"], df["Weights random"], label='Random')
    plt.plot(df["Train Split"], df["Weights freezed"], label='Freezed')

    plt.legend()
    plt.show()

    print(df["Train Split"])


def run_complex_test(algorithm, dataset_name):
    results = []
    write_result(algorithm, ["Train Split", "Weights freezed", "Weights initialized", "Weights random"])

    for train_split in exp_splits:
        percentage = 0.01 * train_split
        print("running test for: {}%".format(train_split))

        a_s = []
        b_s = []
        c_s = []

        for i in range(repetitions):
            a = run_single_test(algorithm, dataset_name, percentage, True, True)  # load and freeze weights
            b = run_single_test(algorithm, dataset_name, percentage, True, False)  # load weights and train
            c = run_single_test(algorithm, dataset_name, percentage, False, False)  # random initialization

            print("train split:{} model accuracy freezed: {}, initialized: {}, random: {}".format(percentage, a, b, c))

            a_s.append(a)
            b_s.append(b)
            c_s.append(c)

        data = [str(train_split) + "%", np.mean(np.array(a_s)), np.mean(np.array(b_s)), np.mean(np.array(c_s))]
        results.append(data)
        write_result(algorithm, data)


if __name__ == "__main__":
    init(run_complex_test, "test", NGPUS)
