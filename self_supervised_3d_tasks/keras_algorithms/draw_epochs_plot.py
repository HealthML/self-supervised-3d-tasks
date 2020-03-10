import json
from pathlib import Path
import pandas
import glob
import matplotlib.pyplot as plt
import numpy as np

def draw_convergence(name, data, data_name, algorithm="exemplar",min_max_avg="avg", batch_size=32, repetitions=3, epochs=12, dataset_name="kaggle", train3D=False, **kwargs):
    for x,i in data:
        plt.plot(x, label=i)

    plt.title(f"Comparison of {min_max_avg} {data_name} with split {name}%. \n Convergence for {str.upper(algorithm)} on {dataset_name} ({'3D' if train3D else '2D'}). \nbatch_size={batch_size},repetitions={repetitions},epochs={epochs}", pad=30)
    plt.legend()
    plt.show()

def get_data_new(args, path, metric="val_loss"):
    data_splits = []
    epochs = args["epochs"]

    for split in args["exp_splits"]:
        data_exp = []
        for exp in ["frozen", "initialized", "random"]:
            data_rep = []
            for filename in (Path(path) / "logs/").glob(f"split{split}{exp}*.log"):
                df = pandas.read_csv(filename)

                # some runs are missing, break here
                if len(df) < epochs:
                    return np.stack(data_splits)

                # otherwise continue
                data_rep.append(df[metric].to_numpy())

            if len(data_rep) == 0:
                return np.stack(data_splits)

            data_exp.append(np.stack(data_rep))
        data_splits.append(np.stack(data_exp))

    return np.stack(data_splits)

def get_data(args, path, metric="val_loss"):
    data_splits = []
    epochs = args["epochs"]

    for index in range(len(args["exp_splits"])):
        data_exp = []
        for s in ["frozen", "initialized", "random"]:
            data_rep = []
            for filename in (Path(path) / "logs/").glob(f"{s}*.log"):
                df = pandas.read_csv(filename)
                subset = df.iloc[epochs * index:epochs * index + epochs]

                # some runs are missing
                if len(subset) < epochs:
                    return np.stack(data_splits)

                data_rep.append(subset[metric].to_numpy())
            data_exp.append(np.stack(data_rep))
        data_splits.append(np.stack(data_exp))

    return np.stack(data_splits)

def draw_old_plots(path):
    data_name = "val_weighted_dice_coefficient"
    try:
        config = list(Path(path).glob("*.json"))[0]
        with open(config, mode="r") as file:
            args = json.load(file)
        print(args)
        data = get_data(args, path, data_name)
        for x, y in zip(data, args["exp_splits"][:len(data)]):
            draw_convergence(y, zip(np.average(x, axis=1), ["frozen", "initialized", "random"]), data_name=data_name,
                             **args)
    except IndexError as e:
        raise FileNotFoundError("No JSON file found in provided directory.")

def draw_new_plots(path, data_name):
    try:
        config = list(Path(path).glob("*.json"))[0]
        with open(config, mode="r") as file:
            args = json.load(file)
        print(args)
        data = get_data_new(args, path, data_name)
        for x, y in zip(data, args["exp_splits"][:len(data)]):
            draw_convergence(y, zip(np.average(x, axis=1), ["frozen", "initialized", "random"]), data_name=data_name,
                             **args)
    except IndexError as e:
        raise FileNotFoundError("No JSON file found in provided directory.")

if __name__ == "__main__":
    epath = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/exemplar_pancreas3d/weights-improvement-216_test_1/"
    rpath = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/rotation_pancreas3d_3/weights-improvement-233_test_8/"

    path = rpath

    draw_new_plots(path, "weighted_dice_coefficient")