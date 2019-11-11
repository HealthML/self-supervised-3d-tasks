"""Exemplar implementation with triplet semihard loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras

import trainer

from self_supervised_3d_tasks import utils
from self_supervised_3d_tasks.models.utils import get_net
from self_supervised_3d_tasks.trainer import make_estimator

FLAGS = tf.flags.FLAGS

def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x

def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(outputs)

    return output

class CPCLayer(keras.engine.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = tf.math.reduce_mean(y_encoded * preds, axis=-1)
        dot_product = tf.math.reduce_mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = tf.math.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

def apply_model(image_fn,  # pylint: disable=missing-docstring
                is_training,
                make_signature=False):
    # Image tensor needs to be created lazily in order to satisfy tf-hub
    # restriction: all tensors should be created inside tf-hub helper function.
    data = image_fn()
    encoded = data["encoded"]
    encoded_pred = data['encoded_pred']

    # terms = images.shape[0]
    predict_terms = encoded_pred.shape[0]

    # using a generic encoder
    encoder_model = get_net("cpc_encoder")

    ##################
    ##################
    ##################
    ##################

    # learning_rate = FLAGS.get_flag_value('learning_rate', 1e-4)
    code_size = FLAGS.get_flag_value('code_size', 128)

    # add specific Layers for CPC
    # x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(encoded)

    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    # y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(encoded_pred)

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
        hub.add_signature(inputs={'encoded': data["encoded"], 'encoded_pred': data['encoded_pred']}, outputs=dot_product_probs)
        hub.add_signature(
            name='representation',
            inputs={'encoded': data["encoded"]},
            outputs=context)

    return dot_product_probs


# TODO: is this really the right shape?
def model_fn(data, mode,serving_input_shape="None,None,None,None,3"):
    """Produces a loss for the exemplar task supervision.

    Args:
      data: Dict of inputs containing, among others, "image" and "label."
      mode: model's mode: training, eval or prediction

    Returns:
      EstimatorSpec
    """

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:

        labels = data['labels']

        with tf.variable_scope('module'):
            image_fn = lambda: data
            dot_product_probs = apply_model(
                image_fn=image_fn,
                is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                make_signature=False)

    else:
        input_shape = utils.str2intlist(serving_input_shape)
        image_fn = lambda: tf.placeholder(  # pylint: disable=g-long-lambda
            shape=input_shape, dtype=tf.float32
        )

        apply_model_function = functools.partial(
            apply_model,
            image_fn=image_fn,
            make_signature=True,
        )

        tf_hub_module_spec = hub.create_module_spec(
            apply_model_function,
            [
                (utils.TAGS_IS_TRAINING, {"is_training": True}),
                (set(), {"is_training": False}),
            ],
            drop_collections=["summaries"],
        )

        tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
        hub.register_module_for_export(tf_hub_module, export_name="module")
        dot_product_probs = tf_hub_module([data['encoded'], data['encoded_pred']])
        return make_estimator(mode, predictions=dot_product_probs)

    eval_metrics = (
        lambda lab, prod: {  # TODO: check if this is correct
            "accuracy": tf.metrics.accuracy(
                labels=lab, predictions=tf.argmax(prod, axis=-1)
            )
        },
        [labels, dot_product_probs],
    )

    loss = keras.losses.binary_crossentropy(dot_product_probs, labels)
    logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)
    return trainer.make_estimator(mode=mode, loss=loss, eval_metrics=eval_metrics, common_hooks=logging_hook)
