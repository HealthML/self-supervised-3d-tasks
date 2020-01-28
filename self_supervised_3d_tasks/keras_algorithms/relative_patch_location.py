from keras import Model, Input
from keras.layers import Flatten, Dense, TimeDistributed
from self_supervised_3d_tasks.custom_preprocessing.relative_patch_location import preprocess_batch
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator


# optionally load this from a config file at some time
data_dir = "/mnt/mpws2019cl1/retinal_fundus/left/max_256/"
data_dim = 192
data_shape = (data_dim, data_dim)
crop_size = 186
patches_per_side = 3
n_channels = 3
code_size = 128
lr = 1e-3
terms = 3
predict_terms = 3
image_size = 40  # this is important to be chosen like the final size of the patches (could auto generate this later on)
test_split = 0.2
img_shape = (image_size, image_size, n_channels)
train_split = 0.7
patch_jitter = 24


def get_training_model():
    model = get_res_net_2d(input_shape=(image_size, image_size, n_channels), classes=patches_per_side**2, architecture="ResNet50", learning_rate=lr)

    return model


def get_training_generators(batch_size, dataset_name):
    def f_train(x, y):  # not using y here, as it gets generated
        return preprocess_batch(x, patches_per_side, patch_jitter)

    def f_val(x, y):
        return preprocess_batch(x, patches_per_side, patch_jitter, False)

    # TODO: move this switch to get_data_generators
    if dataset_name == "ukb_retina":
        train_data, validation_data = get_data_generators(data_dir, train_split=train_split,
                                                          train_data_generator_args={"batch_size": batch_size,
                                                                                     "dim": data_shape,
                                                                                     "n_channels": n_channels,
                                                                                     "pre_proc_func": f_train},
                                                          test_data_generator_args={"batch_size": batch_size,
                                                                                    "dim": data_shape,
                                                                                    "n_channels": n_channels,
                                                                                    "pre_proc_func": f_val})
        return train_data, validation_data
    else:
        raise ValueError("not implemented")


def get_finetuning_generators(batch_size, dataset_name, training_proportion):
    def f_train(x, y):
        return preprocess(resize(x, data_dim), crop_size, patches_per_side, f=preprocess_batch), y

    def f_val(x, y):
        return preprocess(resize(x, data_dim), crop_size, patches_per_side, is_training=False, f=preprocess_batch), y

    # TODO: move this switch to get_data_generators
    if dataset_name == "kaggle_retina":
        gen = KaggleGenerator(batch_size=batch_size, split=training_proportion, shuffle=False,
                              pre_proc_func_train=f_train, pre_proc_func_val=f_val)
        gen_test = KaggleGenerator(batch_size=batch_size, split=1.0-test_split, shuffle=False,
                                   pre_proc_func_train=f_train, pre_proc_func_val=f_val)
        x_test, y_test = gen_test.get_val_data()

        return gen, x_test, y_test
    else:
        raise ValueError("not implemented")


def get_finetuning_model(load_weights, freeze_weights):
    model = get_res_net_2d(input_shape=[63, 63, n_channels], classes=8, architecture="ResNet50", learning_rate=lr)

    if load_weights:
        # loading weights from Julius here, should be your home..
        model.load_weights('/home/Julius.Severin/workspace/self-supervised-transfer-learning/patch_location_retina/'
                               'weights-improvement-1000-0.48.hdf5')

    if freeze_weights:
        # freeze the encoder weights
        model.trainable = False

    layer_in = Input((patches_per_side * patches_per_side,) + img_shape)
    layer_out = TimeDistributed(model)(layer_in)

    x = Flatten()(layer_out)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(5, activation="sigmoid")(x)

    model = Model(inputs=layer_in, outputs=x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model