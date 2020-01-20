from keras.layers import TimeDistributed

from self_supervised_3d_tasks.algorithms import patch_utils
from self_supervised_3d_tasks.keras_models.fully_connected import fully_connected

from self_supervised_3d_tasks.keras_models.res_net_2d import get_res_net_2d

dim=(192, 192) # ?? use patch size here
n_channels=3
lr=1e-3
embed_dim=1000
architecture="ResNet50"

def get_training_model():
    model = get_res_net_2d(input_shape=[*dim, n_channels], classes=embed_dim, architecture=architecture,
                           learning_rate=lr)
    model = TimeDistributed(model)
    model = fully_connected(model)

    return model

def get_training_generators(batch_size, dataset_name):
    # TODO: add permutation preprocessing
    perms, num_classes = patch_utils.load_permutations()

def get_finetuning_generators(batch_size, dataset_name, training_proportion):
    pass

def get_finetuning_model(load_weights, freeze_weights):
    pass