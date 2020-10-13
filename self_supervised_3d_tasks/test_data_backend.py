from self_supervised_3d_tasks.data.kaggle_retina_data import get_kaggle_generator, get_kaggle_cross_validation
from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.data.numpy_2d_loader import Numpy2DLoader
from self_supervised_3d_tasks.data.segmentation_task_loader import SegmentationGenerator3D, PatchSegmentationGenerator3D
import numpy as np


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
            batch_size, f_train, f_val, train_split, data_generator=SegmentationGenerator3D, **kwargs,
        )
    elif dataset_name == 'brats' or dataset_name == 'ukb3d':
        return get_dataset_regular_train(
            batch_size, f_train, f_val, train_split, data_generator=PatchSegmentationGenerator3D, **kwargs,
        )
    elif dataset_name == "pancreas2d":
        return get_dataset_regular_train(
            batch_size, f_train, f_val, train_split, data_generator=Numpy2DLoader, **kwargs,
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
    elif dataset_name == 'brats' or dataset_name == 'ukb3d':
        gen_test = get_dataset_regular_test(
            batch_size, f_test, data_generator=PatchSegmentationGenerator3D, **kwargs
        )
    elif dataset_name == "pancreas2d":
        gen_test = get_dataset_regular_test(
            batch_size, f_test, data_generator=Numpy2DLoader, **kwargs,
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
        assert dataset_name == "kaggle_retina", "CV only implemented for kaggle so far"

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
