import json
from pathlib import Path
import pandas
import glob
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import numpy as np
from math import inf

def get_brats_data():
    names = ["cpc", "jigsaw", "rotation", "rpl", "exemplar", "baseline"]
    splits = [10, 25, 50, 100]
    data = [[0.76, 0.78, 0.82, 0.91],
            [0.71, 0.76, 0.81, 0.90],
            [0.74, 0.77, 0.80, 0.90],
            [0.74, 0.75, 0.80, 0.91],
            [0.75, 0.77, 0.80, 0.91],
            [0.42, 0.58, 0.72, 0.88]]

    return names, data, splits

def neighbour_smoothing(values, epochs, neighbour_count = 2):
    smoothed_values = []
    smoothed_epochs = []
    for index in range(int(len(values) / (neighbour_count))):
        smoothed_values.append(np.mean(values[index*(neighbour_count): index*(neighbour_count) + neighbour_count]))
        smoothed_epochs.append(epochs[index * (neighbour_count)])
    return smoothed_values, smoothed_epochs


def get_metric_over_split(args, path, metric):
    filename = Path(path) / "results.csv"
    df = pandas.read_csv(filename)

    def percentage_string_to_int(string):
        return int(string[:-1])

    splits = df["Train Split"].to_numpy()
    splits = map(percentage_string_to_int, splits)

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

        values = df[metric].to_numpy()
        epochs = range(len(values))
        values_per_rep.append(values)
    values = np.mean(np.array(values_per_rep), axis=0)
    return epochs[::nth_epoch], values[::nth_epoch]


def draw_curve(x_data, y_data, label):
    plt.plot(x_data, y_data, label=label, marker="o", linewidth=3, markersize=15)

    return


def draw_epoch_plot(paths, data_names, metric, nth_epoch, neighbour_count=1, prefix="2d", metric_name=None):
    plt.figure(figsize=(15, 10))
    min_value = inf
    max_value = -inf

    if metric_name is None:
        metric_name = metric.replace("_", " ").title()

    for index, path in enumerate(paths):
        data_name = data_names[index]

        try:
            config = list(Path(path).glob("*.json"))[0]
            with open(config, mode="r") as file:
                args = json.load(file)
            epochs, values = get_metric_over_epochs(args, path, metric, nth_epoch=nth_epoch)
            values, epochs = neighbour_smoothing(values, epochs, neighbour_count)

            draw_curve(epochs, values, data_name)
            min_value = min([min_value, *values])
            max_value = max([max_value, *values])
        except IndexError as e:
            raise FileNotFoundError("No JSON file found in provided directory:" + path)

    plt.legend(loc=2, bbox_to_anchor=(0.65, 0.05 + len(data_names) * 0.08), fontsize=30, borderaxespad=0.)

    plt.ylabel(metric_name, fontsize=26, fontweight='bold')
    plt.xlabel("Epochs", fontsize=26, fontweight='bold')

    plt.yticks(np.arange(round(min_value, 1), round(max_value, 1) + 0.04, 0.02), fontsize=24)
    plt.xticks(fontsize=24)

    plt.grid()
    plt.savefig("plots/" + '_'.join(["epochs", prefix, *data_names]) + ".pdf", bbox_inches='tight')
    plt.show()


def draw_train_split_plot(paths, data_names, metric, skips = [], prefix="2d", metric_name=None):
    plt.figure(figsize=(15, 10))
    min_value = inf
    max_value = -inf

    if metric_name is None:
        metric_name = metric.replace("_", " ").title()

    for index, path in enumerate(paths):
        data_name = data_names[index]
        path = str(Path(path).expanduser())

        config = list(Path(path).glob("*.json"))[0]
        with open(config, mode="r") as file:
            args = json.load(file)
        print(path)
        splits, values = get_metric_over_split(args, path, metric)
        for skip in skips:
            try:
                skip_index = splits.index(skip)
                splits.pop(skip_index)
                values.pop(skip_index)
            except ValueError as e:
                print(data_name + " does not have split " + str(skip))
        draw_curve(splits, values, data_name)

        min_value = min([min_value, *values])
        max_value = max([max_value, *values])

    plt.legend(loc=2, bbox_to_anchor=(0.65, 0.05 + len(data_names) * 0.08), fontsize=30, borderaxespad=0.)

    plt.ylabel(metric_name, fontsize=26, fontweight='bold')
    plt.xlabel("Percentage of labelled images", fontsize=26, fontweight='bold')

    plt.yticks(np.arange(round(min_value, 1), round(max_value, 1) + 0.1, 0.05), fontsize=24)
    plt.xticks(splits, fontsize=24)

    plt.grid()
    plt.savefig("plots/" + '_'.join(["trainsplit", prefix, *data_names]) + ".pdf", bbox_inches='tight')
    plt.show()


def draw_brats_plot():
    plt.figure(figsize=(15, 10))

    min_value = inf
    max_value = -inf

    names, data, splits = get_brats_data()
    for index in range(len(names)):
        values = data[index]
        data_name = names[index]
        draw_curve(splits, values, data_name)

        min_value = min([min_value, *values])
        max_value = max([max_value, *values])

    plt.legend(loc=2, bbox_to_anchor=(0.65, 0.05 + len(names) * 0.08), fontsize=30, borderaxespad=0.)

    plt.ylabel("WT Dice Score", fontsize=26, fontweight='bold')
    plt.xlabel("Percentage of labelled images", fontsize=26, fontweight='bold')

    plt.yticks(np.arange(round(min_value, 1), round(max_value, 1) + 0.1, 0.05), fontsize=24)
    plt.xticks(splits, fontsize=24)

    plt.grid()
    plt.savefig("plots/" + '_'.join(["trainsplit", "brats"]) + ".pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    combined_2d_path = []
    combined_3d_path = []
    combined_labels = []

    cpc_label = "cpc"
    cpc_2d_path = "~/workspace/self-supervised-3d-tasks/cpc_kaggle_retina/weights-250_test_4/"
    cpc_3d_path = "~/workspace/self-supervised-3d-tasks/cpc_pancreas3d_18/weights-400_test_3"
    combined_labels.append(cpc_label)
    combined_2d_path.append(cpc_2d_path)
    combined_3d_path.append(cpc_3d_path)

    jigsaw_label = "jigsaw"
    jigsaw_2d_path = "~/workspace/self-supervised-3d-tasks/jigsaw_kaggle_retina/weights-improvement-929_test_3/"
    jigsaw_3d_path = "~/workspace/self-supervised-3d-tasks/jigsaw_pancreas3d_9/weights-improvement-671_test_13"
    combined_labels.append(jigsaw_label)
    combined_2d_path.append(jigsaw_2d_path)
    combined_3d_path.append(jigsaw_3d_path)

    rotation_label = "rotation"
    rotation_2d_path = "~/workspace/self-supervised-3d-tasks/rotation_kaggle_retina/weights-improvement-837_test_12/"
    rotation_3d_path = "~/workspace/self-supervised-3d-tasks/rotation_pancreas3d_9/weights-improvement-885_test_14"
    combined_labels.append(rotation_label)
    combined_2d_path.append(rotation_2d_path)
    combined_3d_path.append(rotation_3d_path)

    rpl_label = "rpl"
    rpl_2d_path = "~/workspace/self-supervised-3d-tasks/rpl_kaggle_retina/weights-improvement-879_test_16/"
    rpl_3d_path = "~/workspace/self-supervised-3d-tasks/rpl_pancreas3d_9/weights-improvement-936_test_28"
    combined_labels.append(rpl_label)
    combined_2d_path.append(rpl_2d_path)
    combined_3d_path.append(rpl_3d_path)

    exemplar_label = "exemplar"
    exemplar_2d_path = "~/workspace/self-supervised-3d-tasks/exemplar_kaggle_retina/weights-improvement-641_test/"
    exemplar_3d_path = "~/workspace/self-supervised-3d-tasks/exemplar_pancreas3d_42/weights-300_test_3"
    combined_labels.append(exemplar_label)
    combined_2d_path.append(exemplar_2d_path)
    combined_3d_path.append(exemplar_3d_path)

    baseline_label = "baseline"
    baseline_2d_path = "~/workspace/random_test_3/"
    baseline_3d_path = "~/workspace/random_test_3D/"
    combined_labels.append(baseline_label)
    combined_2d_path.append(baseline_2d_path)
    combined_3d_path.append(baseline_3d_path)

    cpc_new_label = "cpc 2D Slices"
    cpc_new_3d_path = "~/workspace/self-supervised-3d-tasks/cpc_pancreas2d_1/weights-250_test"
    combined_labels.append(cpc_new_label)
    combined_3d_path.append(cpc_new_3d_path)

    jig_new_label = "jigsaw 2D Slices"
    jig_new_3d_path = "~/workspace/self-supervised-3d-tasks/jigsaw_pancreas2d_3/weights-improvement-246_test"
    combined_labels.append(jig_new_label)
    combined_3d_path.append(jig_new_3d_path)

    rot_new_label = "rotation 2D Slices"
    rot_new_3d_path = "~/workspace/self-supervised-3d-tasks/rotation_pancreas2d_9/weights-improvement-250_test_3"
    combined_labels.append(rot_new_label)
    combined_3d_path.append(rot_new_3d_path)

    rpl_new_label = "rpl 2D Slices"
    rpl_new_3d_path = "~/workspace/self-supervised-3d-tasks/rpl_pancreas2d_5/weights-improvement-241_test_1"
    combined_labels.append(rpl_new_label)
    combined_3d_path.append(rpl_new_3d_path)

    exp_new_label = "exemplar 2D Slices"
    exp_new_3d_path = "~/workspace/self-supervised-3d-tasks/exemplar_pancreas2d/weights-improvement-235_test"
    combined_labels.append(exp_new_label)
    combined_3d_path.append(exp_new_3d_path)

    baseline_new_label = "baseline 2D Slices"
    baseline_new_3d_path = "~/workspace/random_test_6"
    combined_labels.append(baseline_new_label)
    combined_3d_path.append(baseline_new_3d_path)

    split_2d_metric = "Weights_initialized_qw_kappa_kaggle_avg"
    split_2d_metric_name = "Avg Qw Kappa"
    # draw_train_split_plot(combined_2d_path, combined_labels, split_2d_metric, skips=[1], metric_name=split_2d_metric_name)
    split_3d_metric = "Weights_initialized_dice_avg"
    split_3d_metric_name = "Avg Dice Scores"
    draw_train_split_plot(combined_3d_path, combined_labels, split_3d_metric, skips=[], prefix="3d", metric_name=split_3d_metric_name)

    exit(0)

    epoch_2d_metric = "val_accuracy"
    draw_epoch_plot(combined_2d_path, combined_labels, epoch_2d_metric, nth_epoch=1)
    epoch_3d_metric = "val_weighted_dice_coefficient"
    epoch_3d_metric_name = "Avg Dice Scores"
    draw_epoch_plot(combined_3d_path, combined_labels, epoch_3d_metric, nth_epoch=1, neighbour_count=10, prefix="3d", metric_name=epoch_3d_metric_name)

    draw_brats_plot()

    for index in range(len(combined_3d_path) - 1):
        algorithms_2d = [combined_2d_path[index], combined_2d_path[-1]]
        algorithms_3d = [combined_3d_path[index], combined_3d_path[-1]]
        labels = [combined_labels[index], combined_labels[-1]]
        title = combined_labels[index]

        draw_train_split_plot(algorithms_2d, labels, split_2d_metric, skips=[1], metric_name=split_2d_metric_name)
        draw_epoch_plot(algorithms_2d, labels, epoch_2d_metric, nth_epoch=1)

        draw_train_split_plot(algorithms_3d, labels, split_3d_metric, skips=[25, 50], prefix="3d", metric_name=split_3d_metric_name)
        draw_epoch_plot(algorithms_3d, labels, epoch_3d_metric, nth_epoch=1, neighbour_count=10, prefix="3d", metric_name=epoch_3d_metric_name)