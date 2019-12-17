"""Util functions for representation learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import glob
import os
import re
import shutil

import numpy as np
import tensorflow as tf

INPUT_DATA_STR = "input_data"
IS_TRAINING_STR = "is_training"
REPR_PREFIX_STR = "representation_"
TAGS_IS_TRAINING = ["is_training"]


class Checkpoint(object):
    dir = None
    file = None
    score = None
    path = None

    def __init__(self, path, score):
        self.dir = os.path.dirname(path)
        self.file = os.path.basename(path)
        self.score = score
        self.path = path


class BestCheckpointCopier(tf.estimator.Exporter):
    checkpoints = None
    checkpoints_to_keep = None
    compare_fn = None
    name = None
    score_metric = None
    sort_key_fn = None
    sort_reverse = None

    def __init__(self, name='best_checkpoints', checkpoints_to_keep=5, score_metric='Loss/total_loss',
                 compare_fn=lambda x, y: x.score < y.score, sort_key_fn=lambda x: x.score, sort_reverse=False):
        self.checkpoints = []
        self.checkpoints_to_keep = checkpoints_to_keep
        self.compare_fn = compare_fn
        self.name = name
        self.score_metric = score_metric
        self.sort_key_fn = sort_key_fn
        self.sort_reverse = sort_reverse
        super(BestCheckpointCopier, self).__init__()

    def _copyCheckpoint(self, checkpoint):
        desination_dir = self._destinationDir(checkpoint)
        os.makedirs(desination_dir, exist_ok=True)

        for file in glob.glob(r'{}*'.format(checkpoint.path)):
            self._log('copying {} to {}'.format(file, desination_dir))
            shutil.copy(file, desination_dir)

    def _destinationDir(self, checkpoint):
        return os.path.join(checkpoint.dir, self.name)

    def _keepCheckpoint(self, checkpoint):
        self._log('keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

        self.checkpoints.append(checkpoint)
        self.checkpoints = sorted(self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)

        self._copyCheckpoint(checkpoint)

    def _log(self, statement):
        tf.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

    def _pruneCheckpoints(self, checkpoint):
        destination_dir = self._destinationDir(checkpoint)

        for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
            self._log('removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

            old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
            for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
                self._log('removing old checkpoint file {}'.format(file))
                os.remove(file)

        self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

    def _score(self, eval_result):
        return float(eval_result[self.score_metric])

    def _shouldKeep(self, checkpoint):
        return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        self._log('export checkpoint {}'.format(checkpoint_path))

        score = self._score(eval_result)
        checkpoint = Checkpoint(path=checkpoint_path, score=score)

        if self._shouldKeep(checkpoint):
            self._keepCheckpoint(checkpoint)
            self._pruneCheckpoints(checkpoint)
        else:
            self._log('skipping checkpoint {}'.format(checkpoint.path))


def adaptive_pool(inp, num_target_dimensions=9000, mode="adaptive_max"):
    """Adaptive pooling layer.

       This layer performs adaptive pooling, such that the total
       dimensionality of output is not bigger than num_target_dimension

    Args:
       inp: input tensor
       num_target_dimensions: maximum number of output dimensions
       mode: one of {"adaptive_max", "adaptive_avg", "max", "avg"}

    Returns:
      Result of the pooling operation

    Raises:
      ValueError: mode is unexpected.
    """

    size, _, k = inp.get_shape().as_list()[1:]
    if mode in ["adaptive_max", "adaptive_avg"]:
        if mode == "adaptive_max":
            pool_fn = tf.nn.fractional_max_pool
        else:
            pool_fn = tf.nn.fractional_avg_pool

        # Find the optimal target output tensor size
        target_size = (num_target_dimensions / float(k)) ** 0.5
        if (abs(num_target_dimensions - k * np.floor(target_size) ** 2) <
                abs(num_target_dimensions - k * np.ceil(target_size) ** 2)):
            target_size = max(np.floor(target_size), 1.0)
        else:
            target_size = max(np.ceil(target_size), 1.0)

        # Get optimal stride. Subtract epsilon to ensure correct rounding in
        # pool_fn.
        stride = size / target_size - 1.0e-5

        # Make sure that the stride is valid
        stride = max(stride, 1)
        stride = min(stride, size)

        result = pool_fn(inp, [1, stride, stride, 1])[0]
    elif mode in ["max", "avg"]:
        if mode == "max":
            pool_fn = tf.contrib.layers.max_pool2d
        else:
            pool_fn = tf.contrib.layers.avg_pool2d
        total_size = float(np.prod(inp.get_shape()[1:].as_list()))
        stride = int(np.ceil(np.sqrt(total_size / num_target_dimensions)))
        stride = min(max(1, stride), size)

        result = pool_fn(inp, kernel_size=stride, stride=stride)
    else:
        raise ValueError("Not supported %s pool." % mode)

    return result


def append_multiple_rows_to_csv(dictionaries, csv_path):
    """Writes multiples rows to csv file from a list of dictionaries.

    Args:
      dictionaries: a list of dictionaries, mapping from csv header to value.
      csv_path: path to the result csv file.
    """

    keys = set([])
    for d in dictionaries:
        keys.update(d.keys())

    if not tf.gfile.Exists(csv_path):
        with tf.gfile.Open(csv_path, "w") as f:
            writer = csv.DictWriter(f, sorted(keys))
            writer.writeheader()
            f.flush()

    with tf.gfile.Open(csv_path, "a") as f:
        writer = csv.DictWriter(f, sorted(keys))
        writer.writerows(dictionaries)
        f.flush()


def concat_dicts(dict_list):
    """Given a list of dicts merges them into a single dict.

    This function takes a list of dictionaries as an input and then merges all
    these dictionaries into a single dictionary by concatenating the values
    (along the first axis) that correspond to the same key.

    Args:
      dict_list: list of dictionaries

    Returns:
      d: merged dictionary
    """
    d = collections.defaultdict(list)
    for e in dict_list:
        for k, v in e.items():
            d[k].append(v)
    for k in d:
        d[k] = tf.concat(d[k], axis=0)
    return d


def str2intlist(s, repeats_if_single=None):
    """Parse a config's "1,2,3"-style string into a list of ints.

    Args:
      s: The string to be parsed, or possibly already an int.
      repeats_if_single: If s is already an int or is a single element list,
                         repeat it this many times to create the list.

    Returns:
      A list of integers based on `s`.
    """
    if isinstance(s, int):
        result = [s]
    else:
        result = [int(i.strip()) if i != "None" else None
                  for i in s.split(",")]
    if repeats_if_single is not None and len(result) == 1:
        result *= repeats_if_single
    return result


def tf_apply_to_image_or_images(fn, image_or_images):
    """Applies a function to a single image or each image in a batch of them.

    Args:
      fn: the function to apply, receives an image, returns an image.
      image_or_images: Either a single image, or a batch of images.

    Returns:
      The result of applying the function to the image or batch of images.

    Raises:
      ValueError: if the input is not of rank 3 or 4.
    """
    print("GET_SHAPE", image_or_images)
    static_rank = len(image_or_images.get_shape().as_list())
    if static_rank == 3:  # A single image: HWC
        return fn(image_or_images)
    elif static_rank == 4:  # A batch of images: BHWC
        return tf.map_fn(fn, image_or_images)
    elif static_rank > 4:  # A batch of images: ...HWC
        input_shape = image_or_images.get_shape().as_list()[0:-3]
        h, w, c = image_or_images.get_shape().as_list()[-3:]
        image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
        image_or_images = tf.map_fn(fn, image_or_images)
        return tf.reshape(image_or_images, input_shape + image_or_images.get_shape().as_list()[-3:])
    else:
        raise ValueError("Unsupported image rank: %d" % static_rank)


def tf_apply_to_scan_or_scans(fn, scan_or_scans):
    """Applies a function to a single image or each image in a batch of them.

    Args:
      fn: the function to apply, receives an image, returns an image.
      scan_or_scans: Either a single image, or a batch of images.

    Returns:
      The result of applying the function to the image or batch of images.

    Raises:
      ValueError: if the input is not of rank 3 or 4.
    """
    static_rank = len(scan_or_scans.get_shape().as_list())
    if static_rank == 4:  # A single image: HWDC
        return fn(scan_or_scans)
    elif static_rank == 5:  # A batch of images: BHWDC
        return tf.map_fn(fn, scan_or_scans)
    elif static_rank > 5:  # A batch of images: ...HWDC
        input_shape = tf.shape(scan_or_scans)
        h, w, d, c = scan_or_scans.get_shape().as_list()[-4:]
        scan_or_scans = tf.reshape(scan_or_scans, [-1, h, w, d, c])
        scan_or_scans = tf.map_fn(fn, scan_or_scans)
        return tf.reshape(scan_or_scans, input_shape)
    else:
        raise ValueError("Unsupported image rank: %d" % static_rank)


def tf_apply_with_probability(p, fn, x):
    """Apply function `fn` to input `x` randomly `p` percent of the time."""
    return tf.cond(
        tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), p),
        lambda: fn(x),
        lambda: x)


def tf_apply_many_with_probability(ps, functions, x):
    """"Apply function `fn[i]` to input `x` randomly `p[i]` percent of the time."""
    if len(ps) != len(functions):
        raise Exception('lengths do not match')

    print(ps, functions, x)
    print(sum(ps[:0]))
    print(sum(ps[:1]))

    rand = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)

    def _test_i(i):
        return tf.cond(
            tf.less(rand, sum(ps[:i+1])),
            lambda: functions[i](x),
            lambda: _test_i(i + 1) if (i + 1) < len(ps) else x)

    return _test_i(0)


def expand_glob(glob_patterns):
    checkpoints = []
    for pattern in glob_patterns:
        checkpoints.extend(tf.gfile.Glob(pattern))
    assert checkpoints, "There are no checkpoints in " + str(glob_patterns)
    return checkpoints


def get_latest_hub_per_task(hub_module_paths):
    """Get latest hub module for each task.

    The hub module path should match format ".*/hub/[0-9]*/module/.*".
    Example usage:
    get_latest_hub_per_task(expand_glob(["/cns/el-d/home/dune/representation/"
                                         "xzhai/1899361/*/export/hub/*/module/"]))
    returns 4 latest hub module from 4 tasks respectivley.

    Args:
      hub_module_paths: a list of hub module paths.

    Returns:
      A list of latest hub modules for each task.

    """
    task_to_path = {}
    for path in hub_module_paths:
        task_name, module_name = path.split("/hub/")
        timestamp = int(re.findall(r"([0-9]*)/module", module_name)[0])
        current_path = task_to_path.get(task_name, "0/module")
        current_timestamp = int(re.findall(r"([0-9]*)/module", current_path)[0])
        if current_timestamp < timestamp:
            task_to_path[task_name] = path
    return sorted(task_to_path.values())


def get_classification_metrics(tensor_names):
    """Gets classification eval metric on input logits and labels.

    Args:
      tensor_names: a list of tensor names for _metrics input tensors.

    Returns:
      A function computes the metric result, from input logits and labels.
    """

    def _top_k_accuracy(k, labels, logits):
        in_top_k = tf.nn.in_top_k(predictions=logits, targets=labels, k=k)
        return tf.metrics.mean(tf.cast(in_top_k, tf.float32))

    def _metrics(labels, *tensors):
        """Computes the metric from logits and labels.

        Args:
          labels: ground truth labels.
          *tensors: tensors to be evaluated.

        Returns:
          Result dict mapping from the metric name to the list of result tensor and
          update_op used by tf.metrics.
        """
        metrics = {}
        assert len(tensor_names) == len(tensors), "Names must match tensors."
        for i in range(len(tensors)):
            tensor = tensors[i]
            name = tensor_names[i]
            for k in (1, 5):
                metrics["top%d_accuracy_%s" % (k, name)] = _top_k_accuracy(k, labels, tensor)

        return metrics

    return _metrics


def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat([input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(indices=ids, values=tf.ones_like(ground_truth, dtype=tf.float32),
                              dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot


def generalised_dice_loss(prediction, ground_truth, type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    intersect = tf.sparse_reduce_sum(one_hot * prediction, reduction_axes=[0])
    seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\"is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0, generalised_dice_score)
    return 1 - generalised_dice_score, one_hot


def _dice_hard_coe(output, target, smooth=1e-5):
    output = tf.cast(output, dtype=tf.float32)
    target = tf.cast(target, dtype=tf.float32)

    inse = tf.reduce_sum(tf.multiply(output, target))
    l = tf.reduce_sum(output)
    r = tf.reduce_sum(target)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    # hard_dice = tf.reduce_mean(hard_dice)
    hard_dice = tf.metrics.mean(hard_dice)
    return hard_dice


def get_segmentation_metrics(tensor_names, num_classes):
    """Gets classification eval metric on input logits and labels.

    Args:
      tensor_names: a list of tensor names for _metrics input tensors.

    Returns:
      A function computes the metric result, from input logits and labels.
    """

    def _create_segmentation_metrics(ground_truth, predictions, num_classes):
        gt_wt = tf.identity(ground_truth)
        gt_wt = tf.where(tf.equal(2, gt_wt), 1 * tf.ones_like(gt_wt), gt_wt)  # ground_truth_wt[ground_truth_wt == 2]=1
        gt_wt = tf.where(tf.equal(3, gt_wt), 1 * tf.ones_like(gt_wt), gt_wt)  # ground_truth_wt[ground_truth_wt == 3]=1
        pd_wt = tf.identity(predictions)
        pd_wt = tf.where(tf.equal(2, pd_wt), 1 * tf.ones_like(pd_wt), pd_wt)  # predictions_wt[predictions_wt == 2] = 1
        pd_wt = tf.where(tf.equal(3, pd_wt), 1 * tf.ones_like(pd_wt), pd_wt)  # predictions_wt[predictions_wt == 3] = 1
        dice_wt = _dice_hard_coe(pd_wt, gt_wt)
        # tumor core
        gt_tc = tf.identity(ground_truth)
        gt_tc = tf.where(tf.equal(2, gt_tc), 0 * tf.ones_like(gt_tc), gt_tc)  # ground_truth_tc[ground_truth_tc == 2]=0
        gt_tc = tf.where(tf.equal(3, gt_tc), 1 * tf.ones_like(gt_tc), gt_tc)  # ground_truth_tc[ground_truth_tc == 3]=1
        pd_tc = tf.identity(predictions)
        pd_tc = tf.where(tf.equal(2, pd_tc), 0 * tf.ones_like(pd_tc), pd_tc)  # predictions_tc[predictions_tc == 2] = 0
        pd_tc = tf.where(tf.equal(3, pd_tc), 1 * tf.ones_like(pd_tc), pd_tc)  # predictions_tc[predictions_tc == 3] = 1
        dice_tc = _dice_hard_coe(pd_tc, gt_tc)
        # enhancing tumor
        gt_et = tf.identity(ground_truth)
        gt_et = tf.where(tf.equal(1, gt_et), 0 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 1]=0
        gt_et = tf.where(tf.equal(2, gt_et), 0 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 2]=0
        gt_et = tf.where(tf.equal(3, gt_et), 1 * tf.ones_like(gt_et), gt_et)  # ground_truth_et[ground_truth_et == 3]=1
        pd_et = tf.identity(predictions)
        pd_et = tf.where(tf.equal(1, pd_et), 0 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 1] = 0
        pd_et = tf.where(tf.equal(2, pd_et), 0 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 2] = 0
        pd_et = tf.where(tf.equal(3, pd_et), 1 * tf.ones_like(pd_et), pd_et)  # predictions_et[predictions_et == 3] = 1
        dice_et = _dice_hard_coe(pd_et, gt_et)
        # mean IoU of all classes
        IoU = tf.metrics.mean_iou(labels=ground_truth, predictions=predictions, num_classes=num_classes)
        return dice_wt, dice_tc, dice_et, IoU

    def _metrics(labels, *tensors):
        """Computes the metric from logits and labels.

        Args:
          labels: ground truth labels.
          *tensors: tensors to be evaluated.

        Returns:
          Result dict mapping from the metric name to the list of result tensor and
          update_op used by tf.metrics.
        """
        metrics = {}
        assert len(tensor_names) == len(tensors), "Names must match tensors."
        for i in range(len(tensors)):
            tensor = tensors[i]
            name = tensor_names[i]
            dice_wt, dice_tc, dice_et, IoU = _create_segmentation_metrics(labels, tensor, num_classes)
            metrics["dice_wt_%s" % name] = dice_wt
            metrics["dice_tc_%s" % name] = dice_tc
            metrics["dice_et_%s" % name] = dice_et
            metrics["IoU_%s" % name] = IoU

        return metrics

    return _metrics


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, ignore_list=None):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        if ignore_list is not None:
            if name in ignore_list:
                continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def iou_loss(logit, y):
    if len(y.shape) == len(logit.shape):
        y = y[..., -1]
    y = labels_to_one_hot(y, tf.shape(logit)[-1])
    y = tf.sparse_tensor_to_dense(y)
    y_pred_flat = tf.layers.flatten(logit)
    y_true_flat = tf.layers.flatten(y)

    intersection = 2 * tf.reduce_sum(y_pred_flat * y_true_flat, axis=1) + 1.

    denominator = tf.reduce_sum(y_pred_flat, axis=1) + tf.reduce_sum(y_true_flat, axis=1) + 1.

    loss = tf.reduce_mean(intersection / denominator)

    return loss, y
