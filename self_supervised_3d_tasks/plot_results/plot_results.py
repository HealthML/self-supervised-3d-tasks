import json
from pathlib import Path
import pandas
import glob
import matplotlib.pyplot as plt
import numpy as np


def get_metric_over_split(args, path, metric):
    splits = args["exp_splits"]
    filename = Path(path) / "results.csv"
    df = pandas.read_csv(filename)
    try:
        values = df[metric].to_numpy()
    except KeyError as e:
        values = df[metric.replace("initialized", "random")].to_numpy()

    splits, values = zip(*sorted(zip(splits, values)))

    print("splits = ", splits)

    return splits, values


def get_metric_over_epochs(args, path, metric, split=100):
    epoch_count = args["epochs_initialized"]

    epochs = []
    values_per_rep = []
    for filename in (Path(path) / "logs/").glob(f"split{split}*.log"):
        df = pandas.read_csv(filename)

        epochs = df["epoch"].to_numpy()
        values = df[metric].to_numpy()
        values_per_rep.append(values)
    values = np.mean(np.array(values_per_rep), axis=0)
    #np.mean(np.array([old_set, new_set]), axis=0)
    return epochs[::9], values[::9]


def draw_curve(x_data, y, label):
    print("DRAWING", x_data, y, label)
    plt.plot(x_data, y, label=label)
    plt.legend()
    return


def draw_epoch_plot(paths, data_names, metric):
    for index, path in enumerate(paths):
        data_name = data_names[index]

        try:
            config = list(Path(path).glob("*.json"))[0]
            with open(config, mode="r") as file:
                args = json.load(file)
            print("ARGS", args)
            epochs, values = get_metric_over_epochs(args, path, metric)
            draw_curve(epochs, values, data_name)
        except IndexError as e:
            raise FileNotFoundError("No JSON file found in provided directory:" + path)
    plt.show()

def draw_train_split_plot(paths, data_names, metric):
    for index, path in enumerate(paths):
        data_name = data_names[index]

        print(list(Path(path).glob("*")))

        config = list(Path(path).glob("*.json"))[0]
        with open(config, mode="r") as file:
            args = json.load(file)
        print("ARGS", args)
        splits, values = get_metric_over_split(args, path, metric)
        draw_curve(splits, values, data_name)
    plt.show()


# One plot that combines all methods in one plot in few shot learning (the x axis is dataset size like you do now),
# the plot will have all methods’ lines plus the random baseline. [one for 3D and one for 2D]
def plot_combined_train_split(paths, data_names, metric):
    return draw_train_split_plot(paths, data_names, metric)


# One plot for speed of convergence for all methods combined. (the x axis is the number of epochs, the y axis is the metric)
# of the dataset [one for 3D and one for 2D]
def plot_combined_number_epochs(paths, data_names, metric):
    return draw_epoch_plot(paths, data_names, metric)


# One plot for few shot results like the one you’re producing already with varying training dataset size. (10 for 3D, 10 for 2D)
def plot_single_dataset_size():
    return

# One plot for speed of convergence comparing it with the random baseline’s speed of convergence. (10 for 3D, 10 for 2D) (edited)
def plot_single_number_epochs():
    return

if __name__ == "__main__":
    combined_3d_path = []
    combined_labels = []

    cpc_label = "cpc"
    cpc_2d_path = ""
    cpc_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/cpc_pancreas3d_18/weights-400_test_1/"
    combined_labels.append(cpc_label)
    combined_3d_path.append(cpc_3d_path)

    jigsaw_label = "jigsaw"
    jigsaw_2d_path = ""
    jigsaw_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/jigsaw_pancreas3d_9/weights-improvement-671_test_7/"
    combined_labels.append(jigsaw_label)
    combined_3d_path.append(jigsaw_3d_path)

    rotation_label = "rotation"
    rotation_2d_path = ""
    rotation_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/rotation_pancreas3d_9/weights-improvement-885_test_12/"
    combined_labels.append(rotation_label)
    combined_3d_path.append(rotation_3d_path)

    rpl_label = "rpl"
    rpl_2d_path = ""
    rpl_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/rpl_pancreas3d_9/weights-improvement-936_test_27/"
    combined_labels.append(rpl_label)
    combined_3d_path.append(rpl_3d_path)

    exemplar_label = "exemplar"
    exemplar_2d_path = ""
    exemplar_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/exemplar_pancreas3d_42/weights-300_test_1/"
    combined_labels.append(exemplar_label)
    combined_3d_path.append(exemplar_3d_path)

    baseline_label = "baseline"
    baseline_2d_path = ""
    baseline_3d_path = "/home/Julius.Severin/self-supervised-3d-taks/random_test/"
    combined_labels.append(baseline_label)
    combined_3d_path.append(baseline_3d_path)

    epoch_metric = "val_dice_class_1"
    split_metric = "Weights_initialized_dice_pancreas_1_max"
    plot_combined_number_epochs(combined_3d_path, combined_labels, epoch_metric)
    plot_combined_train_split(combined_3d_path, combined_labels, split_metric)
