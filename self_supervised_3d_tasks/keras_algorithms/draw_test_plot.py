import json
from pathlib import Path

import glob
import matplotlib.pyplot as plt
import pandas


def draw_curve(path, name, metric, min_max_avg="avg", batch_size=32, repetitions=3, epochs=12):
    df = pandas.read_csv(path)
    df['split'] = [int(i[:-1]) for i in df['Train Split']]
    df = df.sort_values(by=['split'])

    plt.plot(df["Train Split"], df["Weights_initialized_"+metric+"_"+min_max_avg], label=name + ' - Encoder Trainable')
    plt.plot(df["Train Split"], df["Weights_frozen_"+metric+"_"+min_max_avg], label=name + ' - Encoder Frozen')
    plt.plot(df["Train Split"], df["Weights_random_"+metric+"_"+min_max_avg], label='Random')

    plt.title(f"Comparison of {min_max_avg} {metric} for kaggle retina 2019. \nbatch_size={batch_size},repetitions={repetitions},epochs={epochs}", pad=30)

    plt.legend()
    plt.show()


def draw_convergence():
    x1 = [0.7515,0.7515,0.7479,0.7515,0.7503,0.7527,0.7636,0.8303,0.8861,0.8897]
    x2 = [0.8715,0.8861,0.8836,0.8788,0.8812,0.8776,0.8921,0.8521,0.8885,0.8897]

    plt.plot(x2, label="Weights copied from Jigsaw")
    plt.plot(x1, label="Weights randomly initialized")

    plt.title("Comparison of accuracy scores after n epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/cpc_kaggle_retina_13/weights-improvement-004_test_1/results.csv"
    # draw_convergence()

    args = {}
    for filename in glob.glob(str(Path(path).parent) + "/*.json"):
        with open(filename, "r") as file:
            args = json.load(file)
        break

    draw_curve(path, args["algorithm"], "qw_kappa_kaggle", "avg", args["batch_size"],  args["repetitions"],  args["epochs"])
