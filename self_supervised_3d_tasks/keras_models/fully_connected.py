from keras.layers import Conv2D, BatchNormalization, Dropout, Dense


def fully_connected(
        inputs, num_classes=100, rate=0.5, is_training=True
):
    """Two layers fully connected network copied from Alexnet fc7-fc8."""
    net = inputs

    net = Dense(4608, activation="relu")(net)
    net = BatchNormalization(momentum=0.997, epsilon=1e-5)(net)
    net = Dense(4096, activation="relu")(net)
    net = BatchNormalization(momentum=0.997, epsilon=1e-5)(net)

    if is_training:
        net = Dropout(rate=rate)(net)

    net = Dense(num_classes, activation="softmax")(net)

    return net


def fully_connected_old(
        inputs, num_classes=100, keep_prob=0.5, is_training=True
):
    """Two layers fully connected network copied from Alexnet fc7-fc8."""
    net = inputs
    _, _, w, _ = net.get_shape().as_list()
    net = Conv2D(
        filters=4096,
        kernel_size=w,
        padding="same",
        activation="relu"
    )(net)
    net = BatchNormalization(momentum=0.997, epsilon=1e-5, fused=None, training=is_training)(net)

    if is_training:
        net = Dropout(keep_prob=keep_prob)(net)

    net = Conv2D(
        filters=num_classes,
        kernel_size=1,
        padding="same",
        activation="linear"
    )(net)

    return net
