import json
import shutil
import struct
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet152
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2
from tensorflow.keras.layers import Dense, Flatten

from self_supervised_3d_tasks.free_gpu_check import aquire_free_gpus
from self_supervised_3d_tasks.ifttt_notify_me import shim_outputs, Tee
from self_supervised_3d_tasks.keras_models.fully_connected import fully_connected_big
from self_supervised_3d_tasks.keras_models.unet import downconv_model
from self_supervised_3d_tasks.keras_models.unet3d import downconv_model_3d


def init(f, name="training", NGPUS=1):
    config_filename = Path(__file__).parent / "config.json"

    if (len(sys.argv)) > 1:
        config_filename = sys.argv[1]

    with open(config_filename, "r") as file:
        args = json.load(file)
        args["root_config_file"] = config_filename

    if "log_level" in args:
        tf.get_logger().setLevel(args["log_level"])
    else:
        tf.get_logger().setLevel("ERROR")

    print("###########################################")
    print("{} {} with parameters: ".format(name, args))
    print("###########################################")

    aquire_free_gpus(amount=NGPUS, **args)
    (
        c_stdout,
        c_stderr,
    ) = shim_outputs()  # I redirect stdout / stderr to later inform us about errors

    with redirect_stdout(
            Tee(c_stdout, sys.stdout)
    ):  # needed to actually capture stdout
        with redirect_stderr(
                Tee(c_stderr, sys.stderr)
        ):  # needed to actually capture stderr
            f(**args)


def get_prediction_model(name, in_shape):
    if name == "big_fully":
        input_l = Input(in_shape)
        output_l = fully_connected_big(input_l)
        model = Model(input_l, output_l)
    else:
        raise ValueError("model " + name + " not found")

    return model


def apply_prediction_model(
        input_shape,
        n_prediction_layers=2,
        dim_prediction_layers=1024,
        prediction_architecture=None,
        **kwargs
):
    if prediction_architecture is not None:
        model = get_prediction_model(prediction_architecture, input_shape)
    else:
        layer_in = Input(input_shape)
        x = layer_in

        for i in range(n_prediction_layers):
            x = Dense(dim_prediction_layers, activation="relu")(x)

        x = Dense(1, activation="relu")(x)
        model = Model(inputs=layer_in, outputs=x)

    return model


def get_encoder_model(name, in_shape, pooling):
    if name == "InceptionV3":
        model = InceptionV3(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    elif name == "ResNet50":
        model = ResNet50(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    elif name == "ResNet50V2":
        model = ResNet50V2(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    elif name == "ResNet101":
        model = ResNet101(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    elif name == "ResNet101V2":
        model = ResNet101V2(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    elif name == "ResNet152":
        model = ResNet152(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    elif name == "InceptionResNetV2":
        model = InceptionResNetV2(
            include_top=False, input_shape=in_shape, weights=None, pooling=pooling
        )
    else:
        raise ValueError("model " + name + " not found")

    return model


def get_encoder_model_3d(name, in_shape):
    raise ValueError("model " + name + " not found")


def apply_encoder_model_3d(
        input_shape,
        code_size,
        num_layers=4,
        pooling="max",
        encoder_architecture=None,
        **kwargs
):
    if encoder_architecture is not None:
        model = get_encoder_model_3d(encoder_architecture, input_shape)
    else:
        model, _ = downconv_model_3d(
            input_shape, num_layers=num_layers, pooling=pooling
        )

    x = Flatten()(model.outputs[0])
    x = Dense(code_size)(x)

    enc_model = Model(model.inputs[0], x, name="encoder")
    return enc_model


def apply_encoder_model(
        input_shape,
        code_size,
        num_layers=4,
        pooling="max",
        encoder_architecture=None,
        **kwargs
):
    if encoder_architecture is not None:
        model = get_encoder_model(encoder_architecture, input_shape, pooling)
    else:
        model, _ = downconv_model(input_shape, num_layers=num_layers, pooling=pooling)

    x = Flatten()(model.outputs[0])
    x = Dense(code_size)(x)

    enc_model = Model(model.inputs[0], x, name="encoder")
    return enc_model


def load_permutations_3d(
        permutation_path=str(
            Path(__file__).parent.parent / "permutations" / "permutations3d_100_max.bin"
        ),
):
    with open(permutation_path, "rb") as f:
        int32_size = 4
        s = f.read(int32_size * 2)
        [num_perms, c] = struct.unpack("<ll", s)
        perms = []
        for _ in range(num_perms * c):
            s = f.read(int32_size)
            x = struct.unpack("<l", s)
            perms.append(x[0])
        perms = np.reshape(perms, [num_perms, c])

    return perms, num_perms


def load_permutations(
        permutation_path=str(
            Path(__file__).parent.parent / "permutations" / "permutations_100_max.bin"
        ),
):
    """Loads a set of pre-defined permutations."""
    with open(permutation_path, "rb") as f:
        int32_size = 4
        s = f.read(int32_size * 2)
        [num_perms, c] = struct.unpack("<ll", s)
        perms = []
        for _ in range(num_perms * c):
            s = f.read(int32_size)
            x = struct.unpack("<l", s)
            perms.append(x[0])
        perms = np.reshape(perms, [num_perms, c])

    # The bin file used index [1,9] for permutation, updated to [0, 8] for index.
    perms = perms - 1
    return perms, num_perms


def get_writing_path(working_dir, root_config_file):
    working_dir = str(working_dir)

    i = 1
    while Path(working_dir).exists():
        if i > 1:
            working_dir = working_dir[: -len(str(i - 1))]
        else:
            working_dir += "_"

        working_dir += str(i)
        i += 1

    Path(working_dir).mkdir()
    print("writing to: " + working_dir)
    shutil.copy2(root_config_file, working_dir)

    return Path(working_dir)
