from tensorflow.keras.layers import BatchNormalization, Dropout, Dense


def fully_connected(
        inputs, num_classes=100, rate=0.5
):
    net = inputs

    net = Dense(512, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=rate)(net)

    net = Dense(512, activation="relu")(net)
    net = BatchNormalization()(net)

    net = Dense(num_classes, activation="softmax")(net)

    return net


def fully_connected_big(inputs, rate=0.5):
    net = inputs

    net = Dense(2048, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=rate)(net)

    net = Dense(1024, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=rate)(net)

    net = Dense(1, activation="relu")(net)
    # output for retina kaggle (positive)

    return net
