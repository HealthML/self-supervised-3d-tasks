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
