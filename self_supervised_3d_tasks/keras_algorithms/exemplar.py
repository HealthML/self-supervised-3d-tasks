from os.path import expanduser

import numpy as np
from self_supervised_3d_tasks.custom_preprocessing.preprocessing_exemplar import preprocessing_exemplar

from self_supervised_3d_tasks.algorithms import patch_utils
from keras.applications import ResNet50
from keras.optimizers import Adam

np.random.seed(0)

from keras.layers import Input, TimeDistributed
from keras.models import Model

from keras.layers.core import Lambda, Flatten, Dense

from keras.engine.topology import Layer
from keras import backend as K

working_dir = expanduser("~/workspace/self-supervised-transfer-learning/exemplar_ResNet")
data_path = "/mnt/mpws2019cl1/retinal_fundus/left/max_256"
number_channels = 3
dim = (256, 256)
batch_size = 4
work_dir = working_dir
data_dir = data_path
n_channels = number_channels
dim_3d = False
model_params = {
            "alpha_triplet": 0.2,
            "embedding_size": 10,
            "lr": 0.0006
            }
#TODO Adjust model checkpoint
model_checkpoint= "/".join(working_dir, "")



class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def apply_model(input_shape, alpha_triplet=0.2, embedding_size=10, lr=0.0006):
    network = ResNet50(input_shape=input_shape, include_top=False, weights=None, classes=embedding_size)
    # Define the tensors for the three input images
    input = Input((3, *input_shape), name="anchor_input")
    anchor_input = Lambda(lambda x: x[0, :])(input)
    positive_input = Lambda(lambda x: x[1, :])(input)
    negative_input = Lambda(lambda x: x[2, :])(input)

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=alpha_triplet, name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])

    optimizer = Adam(lr=lr)

    # Connect the inputs with the outputs
    model = Model(inputs=input, outputs=loss_layer)
    model.compile(loss=None, optimizer=optimizer)
    return network, model


def get_training_model():
    return apply_model()[1]


def get_training_preprocessing():
    perms, _ = patch_utils.load_permutations()

    def f_train(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    def f_val(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    return f_train, f_val


def get_finetuning_preprocessing():
    def f_train(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    def f_val(x, y):
        return preprocessing_exemplar(x, y, dim_3d)

    return f_train, f_val


def get_finetuning_layers(load_weights, freeze_weights):
    enc_model, model_full = apply_model()

    if load_weights:
        model_full.load_weights(model_checkpoint)

    if freeze_weights:
        # freeze the encoder weights
        enc_model.trainable = False

    layer_in = Input((*dim, number_channels), name="anchor_input")
    layer_out = TimeDistributed(enc_model)(layer_in)

    x = Flatten()(layer_out)
    return layer_in, x, [enc_model, model_full]