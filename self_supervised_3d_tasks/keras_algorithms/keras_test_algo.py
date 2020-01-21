import csv
import sys
from contextlib import redirect_stdout, redirect_stderr

import matplotlib.pyplot as plt
import numpy as np
import pandas

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import keras_algorithm_list

epochs = 3
repetitions = 5
batch_size = 8
exp_splits = [0.5, 1, 2, 5, 10, 25, 50, 80]  # different train splits to try in %


def run_single_test(algorithm, dataset_name, train_split, load_weights, freeze_weights):
    algorithm_def = keras_algorithm_list[algorithm]

    gen, x_test, y_test = algorithm_def.get_finetuning_generators(batch_size, dataset_name, train_split)
    model = algorithm_def.get_finetuning_model(load_weights, freeze_weights)

    model.fit_generator(generator=gen, epochs=epochs)

    scores = model.evaluate(x_test, y_test)
    print("model accuracy: {}".format(scores[1]))

    return scores[1]


def write_result(name, row):
    with open(name + '_results.csv', 'a') as csvfile:
        result_writer = csv.writer(csvfile, delimiter=',')
        result_writer.writerow(row)


def draw_curve(name):
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

        data = [str(train_split)+"%", np.mean(np.array(a_s)), np.mean(np.array(b_s)), np.mean(np.array(c_s))]
        results.append(data)
        write_result(algorithm, data)


if __name__ == "__main__":
    aquire_free_gpus()
    c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            run_complex_test("jigsaw", "kaggle_retina")
