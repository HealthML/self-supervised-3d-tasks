import sys
import keras

from contextlib import redirect_stdout, redirect_stderr
from os.path import expanduser
from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.keras_algorithms import cpc, jigsaw, relative_patch_location

keras_algorithm_list = {
    "cpc": cpc,
    "jigsaw": jigsaw,
    "rpl": relative_patch_location
}


def train_model(algorithm, dataset_name, epochs=250, batch_size=8):
    working_dir = expanduser("~/workspace/self-supervised-transfer-learning/") + algorithm + "_" + dataset_name
    algorithm_def = keras_algorithm_list[algorithm]

    train_data, validation_data = algorithm_def.get_training_generators(batch_size, dataset_name=dataset_name)
    model = algorithm_def.get_training_model()

    # update after 500 samples
    tb_c = keras.callbacks.TensorBoard(log_dir=working_dir, batch_size=batch_size, update_freq=500)
    mc_c = keras.callbacks.ModelCheckpoint(working_dir + "/weights-improvement-{epoch:03d}.hdf5")
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
    aquire_free_gpus()
    c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            train_model("rpl", "ukb_retina")
