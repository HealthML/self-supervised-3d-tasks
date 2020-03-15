import json
from pathlib import Path
import pandas
import glob
import matplotlib.pyplot as plt
import matplotlib.markers as markers
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

    return list(splits), list(values)


def get_metric_over_epochs(args, path, metric, split=100, nth_epoch=1):
    epoch_count = args["epochs_initialized"]

    epochs = []
    values_per_rep = []
    for filename in (Path(path) / "logs/").glob(f"split{split}*.log"):
        df = pandas.read_csv(filename)

        epochs = df["epoch"].to_numpy()
        values = df[metric].to_numpy()
        values_per_rep.append(values)
    values = np.mean(np.array(values_per_rep), axis=0)
    return epochs[::nth_epoch], values[::nth_epoch]


def draw_curve(x_data, y_data, label):
    plt.plot(x_data, y_data, label=label, marker="o", linewidth=3, markersize=15)

    return


def draw_epoch_plot(paths, data_names, metric, nth_epoch):
    plt.figure(figsize=(15, 10))

    for index, path in enumerate(paths):
        data_name = data_names[index]

        try:
            config = list(Path(path).glob("*.json"))[0]
            with open(config, mode="r") as file:
                args = json.load(file)
            epochs, values = get_metric_over_epochs(args, path, metric, nth_epoch=nth_epoch)
            draw_curve(epochs, values, data_name)
        except IndexError as e:
            raise FileNotFoundError("No JSON file found in provided directory:" + path)

    plt.legend(loc=2, bbox_to_anchor=(0.70, 0.1 + len(data_names) * 0.06), fontsize=30, borderaxespad=0.)

    plt.ylabel(metric.replace("_", " "), fontsize=26, fontweight='bold')
    plt.xlabel("Epochs", fontsize=26, fontweight='bold')

    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    plt.grid()
    plt.savefig("plots/" + '_'.join(["epochs", *data_names]) + ".png")
    plt.show()



def draw_train_split_plot(paths, data_names, metric, skips = []):
    plt.figure(figsize=(15, 10))

    for index, path in enumerate(paths):
        data_name = data_names[index]

        config = list(Path(path).glob("*.json"))[0]
        with open(config, mode="r") as file:
            args = json.load(file)
        splits, values = get_metric_over_split(args, path, metric)
        for skip in skips:
            skip_index = splits.index(skip)
            splits.pop(skip_index)
            values.pop(skip_index)
        draw_curve(splits, values, data_name)

    #plt.title("Title")
    plt.legend(loc=2, bbox_to_anchor=(0.70, 0.1 + len(data_names) * 0.06), fontsize=30, borderaxespad=0.)

    plt.ylabel(metric.replace("_", " "), fontsize=26, fontweight='bold')
    plt.xlabel("Percentage of labelled images", fontsize=26, fontweight='bold')

    plt.yticks(fontsize=24)
    plt.xticks(splits, fontsize=24)

    plt.grid()
    plt.savefig("plots/" + '_'.join(["trainsplit", *data_names]) + ".png")
    plt.show()


if __name__ == "__main__":
    combined_2d_path = []
    combined_3d_path = []
    combined_labels = []

    cpc_label = "cpc"
    cpc_2d_path = ""
    cpc_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/cpc_pancreas3d_18/weights-400_test_1/"
    combined_labels.append(cpc_label)
    combined_2d_path.append(cpc_2d_path)
    combined_3d_path.append(cpc_3d_path)

    jigsaw_label = "jigsaw"
    jigsaw_2d_path = ""
    jigsaw_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/jigsaw_pancreas3d_9/weights-improvement-671_test_7/"
    combined_labels.append(jigsaw_label)
    combined_2d_path.append(jigsaw_2d_path)
    combined_3d_path.append(jigsaw_3d_path)

    rotation_label = "rotation"
    rotation_2d_path = ""
    rotation_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/rotation_pancreas3d_9/weights-improvement-885_test_12/"
    combined_labels.append(rotation_label)
    combined_2d_path.append(rotation_2d_path)
    combined_3d_path.append(rotation_3d_path)

    rpl_label = "rpl"
    rpl_2d_path = ""
    rpl_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/rpl_pancreas3d_9/weights-improvement-936_test_27/"
    combined_labels.append(rpl_label)
    combined_2d_path.append(rpl_2d_path)
    combined_3d_path.append(rpl_3d_path)

    exemplar_label = "exemplar"
    exemplar_2d_path = ""
    exemplar_3d_path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/exemplar_pancreas3d_42/weights-300_test_1/"
    combined_labels.append(exemplar_label)
    combined_2d_path.append(exemplar_2d_path)
    combined_3d_path.append(exemplar_3d_path)

    baseline_label = "baseline"
    baseline_2d_path = ""
    baseline_3d_path = "/home/Julius.Severin/self-supervised-3d-taks/random_test/"
    combined_labels.append(baseline_label)
    combined_2d_path.append(baseline_2d_path)
    combined_3d_path.append(baseline_3d_path)

    split_metric = "Weights_initialized_dice_pancreas_1_max"
    draw_train_split_plot(combined_3d_path, combined_labels, split_metric, skips=[25])

    epoch_metric = "val_dice_class_1"
    draw_epoch_plot(combined_3d_path, combined_labels, epoch_metric, nth_epoch=20)

    for index in range(len(combined_3d_path) - 1):
        algorithms = [combined_3d_path[index], combined_3d_path[-1]]
        labels = [combined_labels[index], combined_labels[-1]]
        title = combined_labels[index]

        draw_train_split_plot(algorithms, labels, split_metric, skips=[25])
        draw_epoch_plot(algorithms, labels, epoch_metric, nth_epoch=20)