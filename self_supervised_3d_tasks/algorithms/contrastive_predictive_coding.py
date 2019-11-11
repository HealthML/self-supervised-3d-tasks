"""Exemplar implementation with triplet semihard loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras

import self_supervised_3d_tasks.trainer as trainer
from self_supervised_3d_tasks.models.utils import get_net


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


def apply_model(image_fn, is_training, code_size=128, make_signature=False):
    # Image tensor needs to be created lazily in order to satisfy tf-hub
    # restriction: all tensors should be created inside tf-hub helper function.
    data = image_fn()
    images = data["encoded"]
    predict_terms = data["pred_terms"]
    terms = data["terms"]

    # using a generic encoder
    encoder_model = get_net()

    ##################
    ##################
    ##################
    ##################

    # learning_rate = FLAGS.get_flag_value('learning_rate', 1e-4)
    image_shape = (images.shape[0], images.shape[0], 3)

    # add specific Layers for CPC
    x_input = keras.layers.Input(
        (terms, image_shape[0], image_shape[1], image_shape[2])
    )
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)

    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input(
        (predict_terms, image_shape[0], image_shape[1], image_shape[2])
    )
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input)

    # CPC layer
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    # cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model
    # cpc_model.compile(
    #     optimizer=keras.optimizers.Adam(lr=learning_rate),
    #     loss='binary_crossentropy',
    #     metrics=['binary_accuracy']
    # )
    # cpc_model.summary()

    ##################
    ##################
    ##################

    if make_signature:
        hub.add_signature(inputs={"images": data["encoded"]}, outputs=dot_product_probs)
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

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        encoded = data["encoded"]
        encoded_pred = data["encoded_pred"]
        labels = data["labels"]
        pred_terms = data["pred_terms"]
        terms = data["terms"]

        with tf.variable_scope("module"):
            image_fn = lambda: data
            dot_product_probs = apply_model(
                image_fn=image_fn,
                is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                make_signature=False,
            )

        loss = keras.losses.binary_crossentropy(dot_product_probs, labels)
        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)
        return trainer.make_estimator(
            mode=mode,
            loss=loss,
            predictions=dot_product_probs,
            common_hooks=logging_hook,
        )
