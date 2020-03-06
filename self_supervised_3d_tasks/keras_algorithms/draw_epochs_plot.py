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


if __name__ == "__main__":
    path = "/home/Noel.Danz/workspace/self-supervised-transfer-learning/exemplar_kaggle_retina_24/weights-improvement-157_test_1"
    # draw_convergence()

    args = {}
    try:
        config = list(Path(path).glob("*.json"))[0]
        with open(config, mode="r") as file:
            args = json.load(file)
        print(args)
        data = get_data(args, path, "val_loss")
        for x, y in zip(data, args["exp_splits"][:len(data)]):
            draw_convergence(y, zip(np.average(x, axis=1), ["frozen", "initialized", "random"]), **args)
    except IndexError as e:
        raise FileNotFoundError("No JSON file found in provided directory.")
