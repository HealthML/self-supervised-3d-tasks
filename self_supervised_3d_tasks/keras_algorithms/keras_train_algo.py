import shutil
from os import path
from os.path import expanduser
from pathlib import Path

import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model

from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms import cpc, jigsaw, relative_patch_location, rotation
from self_supervised_3d_tasks.keras_algorithms.custom_utils import init, get_writing_path

keras_algorithm_list = {
    "cpc": cpc,
    "jigsaw": jigsaw,
    "rpl": relative_patch_location,
    "rotation": rotation
}


def get_dataset(data_dir, batch_size, f_train, f_val, train_val_split):
    train_data, validation_data = get_data_generators(data_dir, train_split=train_val_split,
                                                      train_data_generator_args={"batch_size": batch_size,
                                                                                 "pre_proc_func": f_train},
                                                      test_data_generator_args={"batch_size": batch_size,
                                                                                "pre_proc_func": f_val})

    return train_data, validation_data


def train_model(algorithm, data_dir, dataset_name, root_config_file, epochs=250, batch_size=2, train_val_split=0.9,
                base_workspace="~/workspace/self-supervised-transfer-learning/", **kwargs):
    kwargs["root_config_file"] = root_config_file

    working_dir = get_writing_path(Path(base_workspace).expanduser() / (algorithm + "_" + dataset_name),
                                   root_config_file)
    algorithm_def = keras_algorithm_list[algorithm].create_instance(**kwargs)

    f_train, f_val = algorithm_def.get_training_preprocessing()
    train_data, validation_data = get_dataset(data_dir, batch_size, f_train, f_val, train_val_split)
    model = algorithm_def.get_training_model()
    model.summary()

    # plot_model(model, to_file=expanduser("~/workspace/test.png"), expand_nested=True, show_shapes=True)
    # uncomment if you want to plot the model

    # update after 500 samples
    tb_c = keras.callbacks.TensorBoard(log_dir=str(working_dir), batch_size=batch_size, update_freq=500)
    mc_c = keras.callbacks.ModelCheckpoint(str(working_dir / "weights-improvement-{epoch:03d}.hdf5"), monitor="val_loss",
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
