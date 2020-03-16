from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.python.keras.layers import GlobalAveragePooling2D


def fully_connected(
        inputs, num_classes=100, rate=0.5, include_top=True
):
    net = inputs

    net = Dense(512, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=rate)(net)

    net = Dense(512, activation="relu")(net)
    net = BatchNormalization()(net)

    if include_top:
        net = Dense(num_classes, activation="softmax")(net)

    return net


def simple_multiclass(inputs, dropout_rate=0.5, include_top=True, **kwargs):
    net = inputs
    net = Dense(1024, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=dropout_rate)(net)

    if include_top:
        net = Dense(5, activation='sigmoid')(net)

    return net


def fully_connected_big(inputs, dropout_rate=0.5, include_top=True, **kwargs):
    net = inputs

    net = Dense(2048, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=dropout_rate)(net)

    net = Dense(1024, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dropout(rate=dropout_rate)(net)

    if include_top:
        net = Dense(1, activation="relu")(net)
    # output for retina kaggle (positive)

    return net
