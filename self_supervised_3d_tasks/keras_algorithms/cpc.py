from keras import Model, Input
from keras.layers import Flatten, Dense, TimeDistributed
from self_supervised_3d_tasks.custom_preprocessing.cpc_preprocess import preprocess_grid, preprocess, resize
from self_supervised_3d_tasks.data.data_generator import get_data_generators
from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc
from self_supervised_3d_tasks.models.cnn_baseline import KaggleGenerator

# optionally load this from a config file at some time
data_dir = "/mnt/mpws2019cl1/kaggle_retina/train/resized_384"
data_dim = 384  # original data dim
data_shape = (data_dim, data_dim)
crop_size = 384
split_per_side = 7
n_channels = 3
code_size = 128
lr = 1e-3
terms = 9
predict_terms = 3
image_size = int((crop_size / (split_per_side + 1)) * 2)
test_split = 0.2
img_shape = (image_size, image_size, n_channels)
train_split = 0.7
model_checkpoint = '/home/Julius.Severin/workspace/self-supervised-transfer-learning/cpc_retina/' \
                   'weights-improvement-1000-0.48.hdf5'  # loading weights from Julius here, should be your home..


def get_training_model():
    model, enc_model = network_cpc(image_shape=(image_size, image_size, n_channels), terms=terms,
                                   predict_terms=predict_terms,
                                   code_size=code_size, learning_rate=lr)
    # we get a model that is already compiled
    return model


def get_training_preprocessing(batch_size):
    def f_train(x, y):  # not using y here, as it gets generated
        return preprocess_grid(preprocess(x, crop_size, split_per_side))

    def f_val(x, y):
        return preprocess_grid(preprocess(x, crop_size, split_per_side, is_training=False))

    return f_train, f_val


def get_finetuning_generators(batch_size, dataset_name, training_proportion):
    def f_train(x, y):
        return preprocess(resize(x, data_dim), crop_size, split_per_side, is_training=False), y

    # We are not training CPC here, so is_training = False

    def f_val(x, y):
        return preprocess(resize(x, data_dim), crop_size, split_per_side, is_training=False), y

    # TODO: move this switch to get_data_generators
    gen = KaggleGenerator(batch_size=batch_size, split=training_proportion, shuffle=False,
                          pre_proc_func_train=f_train, pre_proc_func_val=f_val)
    gen_test = KaggleGenerator(batch_size=batch_size, split=1.0 - test_split, shuffle=False,
                               pre_proc_func_train=f_train, pre_proc_func_val=f_val)
    x_test, y_test = gen_test.get_val_data()

    return gen, x_test, y_test


def get_finetuning_model(load_weights, freeze_weights, num_classes=5):
    cpc_model, enc_model = network_cpc(image_shape=img_shape, terms=terms, predict_terms=predict_terms,
                                       code_size=code_size, learning_rate=lr)

    if load_weights:
        cpc_model.load_weights(model_checkpoint)

    if freeze_weights:
        # freeze the encoder weights
        enc_model.trainable = False

    layer_in = Input((split_per_side * split_per_side,) + img_shape)
    layer_out = TimeDistributed(enc_model)(layer_in)

    x = Flatten()(layer_out)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=layer_in, outputs=x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
