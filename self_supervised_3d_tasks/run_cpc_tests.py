from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc
import keras

epochs = 10,
code_size = 128,
lr = 1e-3,
terms = 3,
predict_terms = 3,
image_size = 46,
batch_size = 8
n_channels = 3
crop_size = 186
split_per_side = 7
img_shape = (image_size, image_size, n_channels)

model, enc_model = network_cpc(image_shape=img_shape, terms=terms,
                               predict_terms=predict_terms, code_size=code_size, learning_rate=lr)

# we need so many models
layer_in = [keras.Input(img_shape)] * (split_per_side * split_per_side)
layer_out = [enc_model(x) for x in layer_in]



model.load_weights('my_model_weights.h5')
