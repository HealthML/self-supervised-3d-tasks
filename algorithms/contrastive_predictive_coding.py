"""Exemplar implementation with triplet semihard loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras

from ..utils import get_net


def network_autoregressive(x):

    """ Define the network that integrates information along the sequence """

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name="ar_context")(x)

    return x


def network_prediction(context, code_size, predict_terms):

    """ Define the network mapping context to multiple embeddings """

    outputs = []
    for i in range(predict_terms):
        outputs.append(
            keras.layers.Dense(
                units=code_size, activation="linear", name="z_t_{i}".format(i=i)
            )(context)
        )

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.engine.Layer):

    """ Computes dot product between true and predicted embedding vectors """

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = tf.math.reduce_mean(y_encoded * preds, axis=-1)
        dot_product = tf.math.reduce_mean(
            dot_product, axis=-1, keepdims=True
        )  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = tf.math.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def apply_model(
    image_fn,  # pylint: disable=missing-docstring
    is_training,
    num_outputs,
    learning_rate=1e-4,
    predict_terms=4,
    code_size=128,
    make_signature=False,
):
    # Image tensor needs to be created lazily in order to satisfy tf-hub
    # restriction: all tensors should be created inside tf-hub helper function.
    images = image_fn()

    # using a generic encoder
    net = get_net(num_classes=num_outputs)
    x_encoded, end_points = net(images, is_training)

    ##################
    ##################
    ##################
    ##################

    image_shape = (images.shape[0], images.shape[0], 3)

    # add specific Layers for CPC
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input(
        (predict_terms, image_shape[0], image_shape[1], image_shape[2])
    )
    y_encoded = keras.layers.TimeDistributed(net)(y_input)

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[images, y_input], outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    cpc_model.summary()

    ##################
    ##################
    ##################

    if make_signature:
        hub.add_signature(inputs={"image": images}, outputs=dot_product_probs)
        hub.add_signature(
            name="representation", inputs={"image": images}, outputs=end_points
        )

    return dot_product_probs


def model_fn(data, mode):
    """Produces a loss for the exemplar task supervision.

    Args:
      data: Dict of inputs containing, among others, "image" and "label."
      mode: model's mode: training, eval or prediction

    Returns:
      EstimatorSpec
    """

    images = data["image"]
