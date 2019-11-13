"""Implements fully-supervised model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub

from .. import utils
from .. trainer import make_estimator
from .. datasets import get_num_classes_for_dataset
from .. models.utils import get_net


# FLAGS = tf.flags.FLAGS


def apply_model(
        image_fn,  # pylint: disable=missing-docstring
        is_training,
        num_outputs,
        make_signature=False,
):
    # Image tensor needs to be created lazily in order to satisfy tf-hub
    # restriction: all tensors should be created inside tf-hub helper function.
    images = image_fn()

    net = get_net(num_classes=num_outputs)

    output, end_points = net(images, is_training)

    if make_signature:
        hub.add_signature(inputs={"image": images}, outputs=output)
    return output


class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_checkpoint=None, ignore_list=None):
        self._init_checkpoint = init_checkpoint
        self.ignore_list = ignore_list

    def begin(self):
        tvars = tf.trainable_variables()

        (
            assignment_map,
            initialized_variable_names,
        ) = utils.get_assignment_map_from_checkpoint(
            tvars, self._init_checkpoint, self.ignore_list
        )
        tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(
                "  name = %s, shape = %s%s", var.name, var.shape, init_string
            )


def model_fn(
        data, mode, dataset, checkpoint_dir=None, serving_input_shape="None,None,None,3"
):
    """Produces a loss for the fully-supervised task.

    Args:
      data: Dict of inputs containing, among others, "image" and "label."
      mode: model's mode: training, eval or prediction

    Returns:
      EstimatorSpec
    """
    # TODO: refactor usages
    images = data["image"]

    # In predict mode (called once at the end of training), we only instantiate
    # the model in order to export a tf.hub module for it.
    # This will then make saving and loading much easier down the line.
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_shape = utils.str2intlist(serving_input_shape)
        apply_model_function = functools.partial(
            apply_model,
            image_fn=lambda: tf.placeholder(
                shape=input_shape, dtype=tf.float32
            ),  # pylint: disable=g-long-lambda
            num_outputs=get_num_classes_for_dataset(dataset),
            make_signature=True,
        )
        tf_hub_module_spec = hub.create_module_spec(
            apply_model_function,
            [
                (utils.TAGS_IS_TRAINING, {"is_training": True}),
                (set(), {"is_training": False}),
            ],
        )
        tf_hub_module = hub.Module(tf_hub_module_spec, trainable=False, tags=set())
        hub.register_module_for_export(tf_hub_module, export_name="module")
        predictions = tf_hub_module(images)

        # There is no training happening anymore, only prediciton and model export.
        return make_estimator(mode, predictions=predictions)

    # From here on, we are either in train or eval modes.
    # Create the model in the 'module' name scope so it matches nicely with
    # tf.hub's requirements for import/export later.
    with tf.variable_scope("module"):
        predictions = apply_model(
            image_fn=lambda: images,
            is_training=(mode == tf.estimator.ModeKeys.TRAIN),
            num_outputs=get_num_classes_for_dataset(dataset),
            make_signature=False,
        )

    model_path = None
    if checkpoint_dir:
        checkpoint_dir = str(Path(checkpoint_dir).resolve())
        model_path = tf.train.latest_checkpoint(checkpoint_dir)

    labels = data["label"]

    dice_loss, sparse_one_hot = utils.generalised_dice_loss(
        prediction=tf.nn.softmax(predictions), ground_truth=labels
    )
    dense_one_hot = tf.sparse_tensor_to_dense(sparse_one_hot)

    # dice_loss, dense_one_hot = utils.iou_loss(tf.nn.softmax(predictions), labels)

    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=predictions
    )
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

    # your class weights
    class_weights = tf.constant([[0.1, 100.0, 100.0, 100.0]])
    logits = tf.reshape(predictions, (-1, get_num_classes_for_dataset(dataset)))
    epsilon = tf.constant(value=1e-10)
    logits = logits + epsilon
    label_flat = tf.reshape(labels, (-1, 1))
    labels = tf.reshape(
        tf.one_hot(label_flat, depth=get_num_classes_for_dataset(dataset)),
        (-1, get_num_classes_for_dataset(dataset)),
    )
    softmax = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(
        tf.multiply(labels * tf.log(softmax + epsilon), class_weights),
        reduction_indices=[1],
    )
    weighted_cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # ** computing full loss
    # loss = dice_loss + cross_entropy_loss
    loss = dice_loss
    # loss = cross_entropy_loss
    # loss = weighted_cross_entropy_loss

    # ** evaluation metrics Dice Scores / F1 Scores
    # Gets a metric_fn which evaluates the dice scores for "whole tumor", "tumor core", "enhanced tumor", and avg. IoU
    metrics_fn = utils.get_segmentation_metrics(
        ["predictions"], num_classes=get_num_classes_for_dataset(dataset)
    )

    # A tuple of metric_fn and a list of tensors to be evaluated by TPUEstimator.
    ground_truth = tf.argmax(dense_one_hot, axis=-1, output_type=tf.int32)
    int_predictions = tf.argmax(predictions, axis=-1, output_type=tf.int32)
    eval_metrics_tuple = (metrics_fn, [ground_truth, int_predictions])

    logging_hook = tf.train.LoggingTensorHook(
        {
            "loss": loss,
            "weighted_ce_loss": weighted_cross_entropy_loss,
            "ce_loss": cross_entropy_loss,
            "dice_loss": dice_loss,
        },
        every_n_iter=10,
    )
    if model_path is not None:
        return make_estimator(
            mode,
            loss,
            eval_metrics_tuple,
            common_hooks=[logging_hook],
            train_hooks=[
                RestoreHook(
                    model_path,
                    ignore_list=[
                        "module/decoder/conv3d_5/bias",
                        "module/decoder/conv3d_5/kernel",
                    ],
                )
            ],
        )
    else:
        return make_estimator(
            mode, loss, eval_metrics_tuple, common_hooks=[logging_hook]
        )
