import json
from pathlib import Path
import pandas
import glob
import matplotlib.pyplot as plt
import numpy as np

def draw_convergence(name, data, algorithm="exemplar",min_max_avg="avg", batch_size=32, repetitions=3, epochs=12, dataset_name="kaggle", train3D=False, **kwargs):
    for x,i in data:
        plt.plot(x, label=i)

    plt.title(f"Comparison of {min_max_avg} with split {name}%. Convergence for {algorithm} on {dataset_name} ({'3D' if train3D else '2D'}). \nbatch_size={batch_size},repetitions={repetitions},epochs={epochs}", pad=30)
    plt.legend()
    plt.show()

def get_data(args, path):
    data_splits = []
    epochs = args["epochs"]

    for index in range(len(args["exp_splits"])):
        data_exp = []
        for s in ["frozen", "initialized", "random"]:
            data_rep = []
            for filename in glob.glob(str(Path(path)) + f"/logs/{s}*.log"):
                df = pandas.read_csv(filename)
                subset = df.iloc[epochs*index:epochs*index+epochs]

                # some runs are missing
                if len(subset) < epochs:
                    return np.stack(data_splits)

                data_rep.append(subset["accuracy"].to_numpy())
            data_exp.append(np.stack(data_rep))
        data_splits.append(np.stack(data_exp))

    return np.stack(data_splits)

if __name__ == "__main__":
    path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/jigsaw_pancreas3d_4/weights-improvement-224_test_7/"
    # draw_convergence()

    args = {}
    for filename in glob.glob(str(Path(path)) + "/*.json"):
        with open(filename, "r") as file:
            args = json.load(file)
        break

    data = get_data(args, path)

    for x,y in zip(data, args["exp_splits"][:len(data)]):
        draw_convergence(y, zip(np.average(x, axis=1), ["frozen", "initialized", "random"]), **args)