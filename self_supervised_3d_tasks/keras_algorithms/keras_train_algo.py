from os import path
from os.path import expanduser

import keras
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms import cpc, jigsaw
from self_supervised_3d_tasks.keras_algorithms.custom_utils import init

keras_algorithm_list = {
    "cpc": cpc,
    "jigsaw": jigsaw
}

dataset_dir_list = {
    "kaggle_retina": "/mnt/mpws2019cl1/kaggle_retina/train/resized_384",
    "ukb_retina": "/mnt/mpws2019cl1/retinal_fundus/left/max_256/"
}

train_val_split = 0.9


def get_dataset(dataset_name, batch_size, f_train, f_val):
    if dataset_name not in dataset_dir_list:
        raise ValueError("dataset not implemented")

    data_dir = dataset_dir_list[dataset_name]

    train_data, validation_data = get_data_generators(data_dir, train_split=train_val_split,
                                                      train_data_generator_args={"batch_size": batch_size,
                                                                                 "pre_proc_func": f_train},
                                                      test_data_generator_args={"batch_size": batch_size,
                                                                                "pre_proc_func": f_val})

    return train_data, validation_data


def get_writing_path(algorithm, dataset_name):
    working_dir = expanduser("~/workspace/self-supervised-transfer-learning/") + algorithm + "_" + dataset_name

    i = 0
    while path.exists(working_dir):
        if i > 0:
            working_dir = working_dir[-len(str(i - 1))]
        else:
            working_dir += "_"

        working_dir += str(i)
        i += 1

    print("writing to: " + working_dir)
    return working_dir


def train_model(algorithm, dataset_name, epochs=250, batch_size=16):
    working_dir = get_writing_path(algorithm, dataset_name)
    algorithm_def = keras_algorithm_list[algorithm]

    f_train, f_val = algorithm_def.get_training_preprocessing()
    train_data, validation_data = get_dataset(dataset_name, batch_size, f_train, f_val)
    model = algorithm_def.get_training_model()

    # update after 500 samples
    tb_c = keras.callbacks.TensorBoard(log_dir=working_dir, batch_size=batch_size, update_freq=500)
    mc_c = keras.callbacks.ModelCheckpoint(working_dir + "/weights-improvement-{epoch:03d}.hdf5", monitor="val_loss",
                                           mode="min", save_best_only=True)  # reduce storage space
    callbacks = [tb_c, mc_c]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        callbacks=callbacks
    )


if __name__ == "__main__":
    init(train_model)
