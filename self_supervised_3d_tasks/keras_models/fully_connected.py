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
