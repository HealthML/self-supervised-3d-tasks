from self_supervised_3d_tasks.algorithms.contrastive_predictive_coding import network_cpc

epochs = 10,
code_size = 128,
lr = 1e-3,
terms = 3,
predict_terms = 3,
image_size = 46,
batch_size = 8
n_channels = 3

model, enc_model = network_cpc(image_shape=(image_size, image_size, n_channels), terms=terms,
                               predict_terms=predict_terms, code_size=code_size, learning_rate=lr)

model.load_weights('my_model_weights.h5')
