from keras.layers import Conv2D, BatchNormalization, Dropout


def fully_connected(
        inputs, num_classes=100, keep_prob=0.5, is_training=True
):
    """Two layers fully connected network copied from Alexnet fc7-fc8."""
    net = inputs
    _, _, w, _ = net.get_shape().as_list()
    net = Conv2D(
        net,
        filters=4096,
        kernel_size=w,
        padding="same",
        activation="relu"
    )
    net = BatchNormalization(momentum=0.997, epsilon=1e-5, fused=None, training=is_training)(net)

    if is_training:
        net = Dropout(net, keep_prob=keep_prob)

    net = Conv2D(
        net,
        filters=num_classes,
        kernel_size=1,
        padding="same"
    )

    return net
