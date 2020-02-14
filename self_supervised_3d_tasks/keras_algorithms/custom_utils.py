import sys
from contextlib import redirect_stdout, redirect_stderr

from keras import Model, Input
from keras.applications import ResNet50
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee


def init(f, name="training", NGPUS=1):
    algo = "exemplar"
    dataset = "kaggle_retina"

    if(len(sys.argv)) > 1:
        algo = sys.argv[1]
    if(len(sys.argv)) > 2:
        algo = sys.argv[1]
        dataset = sys.argv[2]

    print("###########################################")
    print("{} {} with data {}".format(name, algo, dataset))
    print("###########################################")

    aquire_free_gpus(NGPUS)
    c_stdout, c_stderr = shim_outputs()  # I redirect stdout / stderr to later inform us about errors

    with redirect_stdout(Tee(c_stdout, sys.stdout)):  # needed to actually capture stdout
        with redirect_stderr(Tee(c_stderr, sys.stderr)):  # needed to actually capture stderr
            f(algo, dataset)


def apply_prediction_model(layer_in, x, multi_gpu = False, lr = 1e-3):
    dim1 = 1024
    dim2 = 1024

    x = Dense(dim1, activation="relu")(x)
    x = Dense(dim2, activation="relu")(x)
    x = Dense(1, activation="relu")(x)

    model = Model(inputs=layer_in, outputs=x)
    if multi_gpu >= 2:
        model = multi_gpu_model(model, gpus=multi_gpu)
    model.compile(
        optimizer=Adam(lr=lr), loss="mse", metrics=["mae"]
    )

    return model


def apply_encoder_model(input_shape, code_size):
    res_net = ResNet50(input_shape=input_shape, include_top=False, weights=None, pooling="max")
    encoder_output = Dense(code_size)(res_net.outputs[0])

    enc_model = Model(res_net.inputs[0], encoder_output, name='encoder')
    return enc_model
