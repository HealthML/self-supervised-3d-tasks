import functools
import os
import random

from self_supervised_3d_tasks.keras_algorithms.callbacks import TerminateOnNaN, NaNLossError, LogCSVWithStart
from self_supervised_3d_tasks.keras_algorithms.losses import weighted_sum_loss, jaccard_distance, \
    weighted_categorical_crossentropy, weighted_dice_coefficient, weighted_dice_coefficient_loss, \
    weighted_dice_coefficient_per_class
from self_supervised_3d_tasks.keras_algorithms.custom_utils import init, model_summary_long, print_flat_summary

import csv
import gc
import tensorflow_addons as tfa

from pathlib import Path

import tensorflow as tf
import numpy as np
from sklearn.metrics import cohen_kappa_score, jaccard_score, accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import Model
from tensorflow.python.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.python.keras.callbacks import CSVLogger

from self_supervised_3d_tasks.data.kaggle_retina_data import get_kaggle_generator, get_kaggle_cross_validation
from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.data.segmentation_task_loader import (
    SegmentationGenerator3D,
)
from self_supervised_3d_tasks.keras_algorithms.custom_utils import (
    apply_prediction_model,
    get_writing_path,
)
from self_supervised_3d_tasks.keras_algorithms.keras_train_algo import (
    keras_algorithm_list,
)

def transform_multilabel_to_continuous(y, threshold):
    assert isinstance(y, np.ndarray), "invalid y"

    y = y > threshold
    y = y.astype(int).sum(axis=1) - 1
    return y


def score_kappa_kaggle(y, y_pred, threshold=0.5):
    y = transform_multilabel_to_continuous(y, threshold)
    y_pred = transform_multilabel_to_continuous(y_pred, threshold)
    return score_kappa(y, y_pred, labels=[0, 1, 2, 3, 4])


def score_kappa(y, y_pred, labels=None):
    if labels is not None:
        return cohen_kappa_score(y, y_pred, labels=labels, weights="quadratic")
    else:
        return cohen_kappa_score(y, y_pred, weights="quadratic")


def score_bin_acc(y, y_pred):
    m = BinaryAccuracy()
    m.update_state(y, y_pred)

    return m.result().numpy()


def score_cat_acc_kaggle(y, y_pred, threshold=0.5):
    y = transform_multilabel_to_continuous(y, threshold)
    y_pred = transform_multilabel_to_continuous(y_pred, threshold)
    return score_cat_acc(y, y_pred)


def score_cat_acc(y, y_pred):
    return accuracy_score(y, y_pred)


def score_jaccard(y, y_pred):
    y = np.argmax(y, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    return jaccard_score(y, y_pred, average="macro")


def score_dice(y, y_pred):
    y = np.argmax(y, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    j = jaccard_score(y, y_pred, average=None)

    return np.average(np.array([(2 * x) / (1 + x) for x in j]))

def score_dice_class(y, y_pred, class_to_predict):
    y = np.argmax(y, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    j = jaccard_score(y, y_pred, average=None)

    return np.array([(2 * x) / (1 + x) for x in j])[class_to_predict]


def get_score(score_name):
    if score_name == "qw_kappa":
        return score_kappa
    elif score_name == "bin_accuracy":
        return score_bin_acc
    elif score_name == "cat_accuracy":
        return score_cat_acc
    elif score_name == "dice":
        return score_dice
    elif score_name == "dice_pancreas_0":
        return functools.partial(score_dice_class, class_to_predict=0)
    elif score_name == "dice_pancreas_1":
        return functools.partial(score_dice_class, class_to_predict=1)
    elif score_name == "dice_pancreas_2":
        return functools.partial(score_dice_class, class_to_predict=2)
    elif score_name == "jaccard":
        return score_jaccard
    elif score_name == "qw_kappa_kaggle":
        return score_kappa_kaggle
    elif score_name == "cat_acc_kaggle":
        return score_cat_acc_kaggle
    else:
        raise ValueError(f"score {score_name} not found")


def make_scores(y, y_pred, scores):
    scores_f = [(x, get_score(x)(y, y_pred)) for x in scores]
    return scores_f


def get_dataset_regular_train(
        batch_size,
        f_train,
        f_val,
        train_split,
        data_generator,
        data_dir_train,
        val_split=0.1,
        train_data_generator_args={},
        val_data_generator_args={},
        **kwargs,
):
    train_split = train_split * (1 - val_split)  # normalize train split

    train_data_generator, val_data_generator, _ = get_data_generators(
        data_generator=data_generator,
        data_path=data_dir_train,
        train_split=train_split,
        val_split=val_split,  # we are eventually not using the full dataset here
        train_data_generator_args={
            **{"batch_size": batch_size, "pre_proc_func": f_train},
            **train_data_generator_args,
        },
        val_data_generator_args={
            **{"batch_size": batch_size, "pre_proc_func": f_val},
            **val_data_generator_args,
        },
        **kwargs,
    )
    return train_data_generator, val_data_generator


def get_dataset_regular_test(
        batch_size,
        f_test,
        data_generator,
        data_dir_test,
        train_data_generator_args={},
        test_data_generator_args={},
        **kwargs,
):
    if "val_split" in kwargs:
        del kwargs["val_split"]

    return get_data_generators(
        data_generator=data_generator,
        data_path=data_dir_test,
        train_data_generator_args={
            **{"batch_size": batch_size, "pre_proc_func": f_test},
            **test_data_generator_args,
        },
        **kwargs,
    )


def get_dataset_kaggle_train_original(
        batch_size,
        f_train,
        f_val,
        train_split,
        csv_file_train,
        data_dir,
        val_split=0.1,
        train_data_generator_args={},
        val_data_generator_args={},
        **kwargs,
):
    train_split = train_split * (1 - val_split)  # normalize train split
    train_data_generator, val_data_generator, _ = get_kaggle_generator(
        data_path=data_dir,
        csv_file=csv_file_train,
        train_split=train_split,
        val_split=val_split,  # we are eventually not using the full dataset here
        train_data_generator_args={
            **{"batch_size": batch_size, "pre_proc_func": f_train},
            **train_data_generator_args,
        },
        val_data_generator_args={
            **{"batch_size": batch_size, "pre_proc_func": f_val},
            **val_data_generator_args,
        },
        **kwargs,
    )
    return train_data_generator, val_data_generator


def get_dataset_kaggle_test(
        batch_size,
        f_test,
        csv_file_test,
        data_dir,
        train_data_generator_args={},  # DO NOT remove
        test_data_generator_args={},
        **kwargs,
):
    if "val_split" in kwargs:
        del kwargs["val_split"]

    return get_kaggle_generator(
        data_path=data_dir,
        csv_file=csv_file_test,
        train_data_generator_args={
            **{"batch_size": batch_size, "pre_proc_func": f_test},
            **test_data_generator_args,
        },
        **kwargs,
    )


def get_data_from_gen(gen):
    print("Loading Test data")

    data = None
    labels = None
    max_iter = len(gen)
    i = 0
    for d, l in gen:
        if data is None:
            data = d
            labels = l
        else:
            data = np.concatenate((data, d), axis=0)
            labels = np.concatenate((labels, l), axis=0)

        print(f"\r{(i * 100.0) / max_iter:.2f}%", end="")
        i += 1
        if i == max_iter:
            break

    print("")

    return data, labels


def get_dataset_train(dataset_name, batch_size, f_train, f_val, train_split, kwargs):
    if dataset_name == "kaggle_retina":
        return get_dataset_kaggle_train_original(
            batch_size, f_train, f_val, train_split, **kwargs
        )
    elif dataset_name == "pancreas3d":
        return get_dataset_regular_train(
            batch_size,
            f_train,
            f_val,
            train_split,
            data_generator=SegmentationGenerator3D,
            **kwargs,
        )
    else:
        raise ValueError("not implemented")


def get_dataset_test(dataset_name, batch_size, f_test, kwargs):
    if dataset_name == "kaggle_retina":
        gen_test = get_dataset_kaggle_test(batch_size, f_test, **kwargs)
    elif dataset_name == "pancreas3d":
        gen_test = get_dataset_regular_test(
            batch_size, f_test, data_generator=SegmentationGenerator3D, **kwargs
        )
    else:
        raise ValueError("not implemented")

    return get_data_from_gen(gen_test)

class StandardDataLoader:
    def __init__(self, dataset_name, batch_size, algorithm_def,
                 **kwargs):
        self.algorithm_def = algorithm_def
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.kwargs = kwargs

    def get_dataset(self, repetition, train_split):
        f_train, f_val = self.algorithm_def.get_finetuning_preprocessing()

        gen_train, gen_val = get_dataset_train(
            self.dataset_name, self.batch_size, f_train, f_val, train_split, self.kwargs
        )

        x_test, y_test = get_dataset_test(self.dataset_name, self.batch_size, f_val, self.kwargs)
        return gen_train, gen_val, x_test, y_test

class CvDataKaggle:
    def __init__(self, dataset_name, batch_size, algorithm_def,
        n_repetitions,
        csv_file,
        data_dir,
        val_split=0.1,
        test_data_generator_args={},
        val_data_generator_args={},
        train_data_generator_args={},
        **kwargs):

        assert dataset_name == "kaggle_retina", "CV only implemented for kaggle"

        f_train, f_val = algorithm_def.get_finetuning_preprocessing()
        self.cv = get_kaggle_cross_validation(data_path=data_dir, csv_file=csv_file,
                                              k_fold=n_repetitions,
                                              train_data_generator_args={
                                                  **{"batch_size": batch_size, "pre_proc_func": f_train},
                                                  **train_data_generator_args,
                                              },
                                              val_data_generator_args={
                                                  **{"batch_size": batch_size, "pre_proc_func": f_val},
                                                  **val_data_generator_args,
                                              },
                                              test_data_generator_args={
                                                  **{"batch_size": batch_size, "pre_proc_func": f_val},
                                                  **test_data_generator_args,
                                              }, **kwargs)
        self.val_split = val_split

    def get_dataset(self, repetition, train_split):
        train_split = train_split * (1 - self.val_split)  # normalize train split

        gen_train, gen_val, gen_test = self.cv.make_generators(test_chunk=repetition, train_split=train_split,
                                                               val_split=self.val_split)

        x_test, y_test = get_data_from_gen(gen_test)
        return gen_train, gen_val, x_test, y_test

def run_single_test(algorithm_def, gen_train, gen_val, load_weights, freeze_weights, x_test, y_test, lr,
                    batch_size, epochs, epochs_warmup, model_checkpoint, scores, loss, metrics, logging_path, kwargs,
                    clipnorm=None, clipvalue=None,
                    model_callback=None):

    def get_optimizer():
        if clipnorm is None and clipvalue is None:
            return Adam(lr=lr)
        elif clipnorm is None:
            return Adam(lr=lr, clipvalue=clipvalue)
        else:
            return Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue)

    if "weighted_dice_coefficient" in metrics:
        metrics.remove("weighted_dice_coefficient")
        metrics.append(weighted_dice_coefficient)

    if "weighted_dice_coefficient_per_class_pancreas" in metrics:
        metrics.remove("weighted_dice_coefficient_per_class_pancreas")

        def dice_class_0(y_true,y_pred):
            return weighted_dice_coefficient_per_class(y_true,y_pred,class_to_predict = 0)
        def dice_class_1(y_true,y_pred):
            return weighted_dice_coefficient_per_class(y_true,y_pred,class_to_predict = 1)
        def dice_class_2(y_true,y_pred):
            return weighted_dice_coefficient_per_class(y_true,y_pred,class_to_predict = 2)

        metrics.append(dice_class_0)
        metrics.append(dice_class_1)
        metrics.append(dice_class_2)

    if load_weights:
        enc_model = algorithm_def.get_finetuning_model(model_checkpoint)
    else:
        enc_model = algorithm_def.get_finetuning_model()

    pred_model = apply_prediction_model(input_shape=enc_model.outputs[0].shape[1:], algorithm_instance=algorithm_def,
                                        **kwargs)

    outputs = pred_model(enc_model.outputs)
    model = Model(inputs=enc_model.inputs[0], outputs=outputs)

    # print_flat_summary(model)

    # TODO: remove debugging
    # plot_model(model, to_file=Path("~/test_architecture.png").expanduser(), expand_nested=True)

    weights = (0.01, 10, 15)
    if loss == "weighted_sum_loss":
        loss = weighted_sum_loss(alpha=0.85, beta=0.15, weights=weights)
    elif loss == "jaccard_distance":
        loss = jaccard_distance
    elif loss == "weighted_dice_loss":
        loss = weighted_dice_coefficient_loss
    elif loss == "weighted_categorical_crossentropy":
        loss = weighted_categorical_crossentropy(weights)

    if epochs > 0:  # testing the scores
        callbacks = [TerminateOnNaN()]

        logging_csv = False
        if logging_path is not None:
            logging_csv = True
            logging_path.parent.mkdir(exist_ok=True, parents=True)
            logger_normal = CSVLogger(str(logging_path), append=False)
            logger_after_warmup = LogCSVWithStart(str(logging_path), start_from_epoch=epochs_warmup, append=True)
        if freeze_weights or load_weights:
            enc_model.trainable = False

        if freeze_weights:
            print(("-" * 10) + "LOADING weights, encoder model is completely frozen")
            if logging_csv:
                callbacks.append(logger_normal)
        elif load_weights:
            assert epochs_warmup < epochs, "warmup epochs must be smaller than epochs"

            print(
                ("-" * 10) + "LOADING weights, encoder model is trainable after warm-up"
            )
            print(("-" * 5) + " encoder model is frozen")

            w_callbacks = list(callbacks)
            if logging_csv:
                w_callbacks.append(logger_normal)

            model.compile(optimizer=get_optimizer(), loss=loss, metrics=metrics)
            model.fit(
                x=gen_train,
                validation_data=gen_val,
                epochs=epochs_warmup,
                callbacks=w_callbacks,
            )
            epochs = epochs - epochs_warmup

            enc_model.trainable = True
            print(("-" * 5) + " encoder model unfrozen")

            if logging_csv:
                callbacks.append(logger_after_warmup)
        else:
            print(("-" * 10) + "RANDOM weights, encoder model is fully trainable")
            if logging_csv:
                callbacks.append(logger_normal)

        # recompile model
        model.compile(optimizer=get_optimizer(), loss=loss, metrics=metrics)
        model.fit(
            x=gen_train, validation_data=gen_val, epochs=epochs, callbacks=callbacks
        )

    model.compile(optimizer=get_optimizer(), loss=loss, metrics=metrics)
    y_pred = model.predict(x_test, batch_size=batch_size)
    scores_f = make_scores(y_test, y_pred, scores)

    if model_callback:
        model_callback(model)

    # cleanup
    del pred_model
    del enc_model
    del model

    algorithm_def.purge()
    K.clear_session()

    for i in range(15):
        gc.collect()

    for s in scores_f:
        print("{} score: {}".format(s[0], s[1]))

    return scores_f


def write_result(base_path, row):
    with open(base_path / "results.csv", "a") as csvfile:
        result_writer = csv.writer(csvfile, delimiter=",")
        result_writer.writerow(row)


class MaxTriesExceeded(Exception):
    def __init__(self, func, *args):
        self.func = func
        if args:
            self.max_tries = args[0]

    def __str__(self):
        return f'Maximum amount of tries ({self.max_tries}) exceeded for {self.func}.'


def try_until_no_nan(func, max_tries=4):
    for _ in range(max_tries):
        try:
            return func()
        except NaNLossError:
            print(f"Encountered NaN-Loss in {func}")
    raise MaxTriesExceeded(func, max_tries)


def run_complex_test(
        algorithm,
        dataset_name,
        root_config_file,
        model_checkpoint,
        epochs_initialized=5,
        epochs_random=5,
        epochs_frozen=5,
        repetitions=2,
        batch_size=8,
        exp_splits=(100, 10, 1),
        lr=1e-3,
        epochs_warmup=2,
        scores=("qw_kappa",),
        loss="mse",
        metrics=("mse",),
        clipnorm=None,
        clipvalue=None,
        do_cross_val=False,
        **kwargs,
):
    if os.path.isdir(model_checkpoint):
        weight_files = list(Path(model_checkpoint).glob("weights-improvement*.hdf5"))
        assert len(weight_files) > 0, "empty directory!"

        weight_files.sort()
        model_checkpoint = str(weight_files[-1])

    kwargs["model_checkpoint"] = model_checkpoint
    kwargs["root_config_file"] = root_config_file
    metrics = list(metrics)  # TODO: this seems unnecessary... but tf expects this to be list or str not tuple -.-

    working_dir = get_writing_path(
        Path(model_checkpoint).expanduser().parent
        / (Path(model_checkpoint).expanduser().stem + "_test"),
        root_config_file,
    )

    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    results = []
    header = ["Train Split"]

    exp_types = []

    if epochs_frozen > 0:
        exp_types.append("Weights_frozen_")

    if epochs_initialized > 0:
        exp_types.append("Weights_initialized_")

    if epochs_random > 0:
        exp_types.append("Weights_random_")

    for exp_type in exp_types:
        for sc in scores:
            for min_avg_max in ["_min", "_avg", "_max"]:
                header.append(exp_type + sc + min_avg_max)

    write_result(working_dir, header)

    if do_cross_val:
        data_loader = CvDataKaggle(dataset_name, batch_size, algorithm_def, n_repetitions=repetitions, **kwargs)
    else:
        data_loader = StandardDataLoader(dataset_name, batch_size, algorithm_def, **kwargs)

    for train_split in exp_splits:
        percentage = 0.01 * train_split
        print("\n--------------------")
        print("running test for: {}%".format(train_split))
        print("--------------------\n")

        a_s = []
        b_s = []
        c_s = []

        for i in range(repetitions):
            logging_base_path = working_dir / "logs"

            # Use the same seed for all experiments in one repetition
            tf.random.set_seed(i)
            np.random.seed(i)
            random.seed(i)

            gen_train, gen_val, x_test, y_test = data_loader.get_dataset(i, percentage)

            if epochs_frozen > 0:
                logging_a_path = logging_base_path / f"split{train_split}frozen_rep{i}.log"
                a = try_until_no_nan(
                    lambda: run_single_test(algorithm_def, gen_train, gen_val, True, True, x_test, y_test, lr,
                                            batch_size, epochs_frozen, epochs_warmup, model_checkpoint, scores, loss,
                                            metrics,
                                            logging_a_path,
                                            kwargs, clipnorm=clipnorm, clipvalue=clipvalue))  # frozen
                a_s.append(a)
            if epochs_initialized > 0:
                logging_b_path = logging_base_path / f"split{train_split}initialized_rep{i}.log"
                b = try_until_no_nan(
                    lambda: run_single_test(algorithm_def, gen_train, gen_val, True, False, x_test, y_test, lr,
                                            batch_size, epochs_initialized, epochs_warmup, model_checkpoint, scores, loss, metrics,
                                            logging_b_path, kwargs, clipnorm=clipnorm, clipvalue=clipvalue))
                b_s.append(b)
            if epochs_random > 0:
                logging_c_path = logging_base_path / f"split{train_split}random_rep{i}.log"
                c = try_until_no_nan(
                    lambda: run_single_test(algorithm_def, gen_train, gen_val, False, False, x_test, y_test, lr,
                                            batch_size, epochs_random, epochs_warmup, model_checkpoint, scores, loss, metrics,
                                            logging_c_path,
                                            kwargs, clipnorm=clipnorm, clipvalue=clipvalue))  # random
                c_s.append(c)

        def get_avg_score(list_abc, index):
            sc = [x[index][1] for x in list_abc]
            return np.mean(np.array(sc))

        def get_min_score(list_abc, index):
            sc = [x[index][1] for x in list_abc]
            return np.min(np.array(sc))

        def get_max_score(list_abc, index):
            sc = [x[index][1] for x in list_abc]
            return np.max(np.array(sc))

        scores_a = []
        scores_b = []
        scores_c = []

        for i in range(len(scores)):
            if epochs_frozen > 0:
                scores_a.append(get_min_score(a_s, i))
                scores_a.append(get_avg_score(a_s, i))
                scores_a.append(get_max_score(a_s, i))

            if epochs_initialized > 0:
                scores_b.append(get_min_score(b_s, i))
                scores_b.append(get_avg_score(b_s, i))
                scores_b.append(get_max_score(b_s, i))

            if epochs_random > 0:
                scores_c.append(get_min_score(c_s, i))
                scores_c.append(get_avg_score(c_s, i))
                scores_c.append(get_max_score(c_s, i))

        data = [str(train_split) + "%"]

        if epochs_frozen > 0:
            data += scores_a

        if epochs_initialized > 0:
            data += scores_b

        if epochs_random > 0:
            data += scores_c

        results.append(data)
        write_result(working_dir, data)


if __name__ == "__main__":
    init(run_complex_test, "test")
