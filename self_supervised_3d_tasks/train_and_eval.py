# pylint: disable=line-too-long
r"""The main script for starting training and evaluation.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division

import argparse
import functools
import json
import logging
import math
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from self_supervised_3d_tasks.errors import MissingFlagsError
from .algorithms.self_supervision_lib import get_self_supervision_model
from .datasets import get_count, get_data
from .utils import BestCheckpointCopier, str2intlist

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# Number of iterations (=training steps) per TPU training loop. Use >100 for
# good speed. This is the minimum number of steps between checkpoints.
TPU_ITERATIONS_PER_LOOP = 500


def train_and_eval(FLAGS):
    """Trains a network on (self_supervised) supervised data."""
    if FLAGS["GPU_Config"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS["GPU_Config"]["CUDA_VISIBLE_DEVICES"]
        gpu_fraction = FLAGS["GPU_Config"].get("GPU_FRACTION", 0.9)
    model_dir = Path(FLAGS["workdir"]).expanduser().resolve()
    use_tpu = FLAGS["use_tpu"]
    batch_size = FLAGS["batch_size"]
    eval_batch_size = FLAGS.get("eval_batch_size", batch_size)
    dataset = FLAGS["dataset"]
    dataset_dir = FLAGS["dataset_dir"]
    preprocessing = FLAGS["preprocessing"]
    epochs = FLAGS["epochs"]
    model_kwargs = FLAGS  # FLAGS.get("model_kwargs", {})

    cluster_master = (
        TPUClusterResolver(tpu=[os.environ["TPU_NAME"]]).get_master() if use_tpu else ""
    )

    configp = tf.ConfigProto()
    configp.gpu_options.allow_growth = True
    configp.allow_soft_placement = True
    configp.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

    config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        tf_random_seed=FLAGS.get("random_seed", None),
        master=cluster_master,
        evaluation_master=cluster_master,
        session_config=configp,
        keep_checkpoint_every_n_hours=FLAGS.get("keep_checkpoint_every_n_hours", 4),
        save_checkpoints_secs=FLAGS.get("save_checkpoints_secs", 600),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=TPU_ITERATIONS_PER_LOOP,
            tpu_job_name=FLAGS.get("tpu_worker_name", ""),
        ),
    )

    # The global batch-sizes are passed to the TPU estimator, and it will pass
    # along the local batch size in the model_fn's `params` argument dict.
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=get_self_supervision_model(FLAGS["task"], model_kwargs=model_kwargs),
        model_dir=model_dir,
        config=config,
        use_tpu=use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )

    if FLAGS.get("run_eval", False):
        eval_mapped = evaluate
        optional_flags = ["val_split", "use_tpu"]
        for flag in optional_flags:
            if FLAGS.get(flag, None):
                eval_mapped = functools.partial(eval_mapped, **{flag: FLAGS[flag]})
        return eval_mapped(
            estimator,
            model_dir,
            dataset,
            preprocessing,
            dataset_dir,
            eval_batch_size=eval_batch_size,
            FLAGS=FLAGS,
        )

    elif FLAGS.get("train_eval", False):
        tace_mapped = train_and_continous_evaluation
        optional_flags = [
            "train_split",
            "val_split",
            "serving_input_shape",
            "serving_input_key",
            "throttle_after",
        ]
        for flag in optional_flags:
            if FLAGS.get(flag, None):
                tace_mapped = functools.partial(tace_mapped, **{flag: FLAGS[flag]})
        tace_mapped(
            estimator,
            dataset,
            preprocessing,
            dataset_dir,
            epochs,
            batch_size,
            use_tpu=use_tpu,
            FLAGS=FLAGS,
        )

    # TRAIN
    else:
        train_mapped = train
        optional_flags = ["train_split"]
        for flag in optional_flags:
            if FLAGS.get(flag, None):
                train_mapped = functools.partial(train_mapped, **{flag: FLAGS[flag]})
        return train_mapped(
            estimator,
            dataset,
            preprocessing,
            dataset_dir,
            epochs,
            batch_size,
            FLAGS=FLAGS,
        )


def train_and_continous_evaluation(
        estimator,
        dataset,
        preprocessing,
        dataset_dir,
        epochs: int,
        batch_size: int,
        train_split="train",
        val_split="val",
        use_tpu=False,
        serving_input_shape="None,None,None,3",
        serving_input_key="image",
        throttle_after=90,
        FLAGS={},
):
    """I train an estimator and evaluate it along the way.

    Args:
        estimator: a tensorflow.estimator.Estimator object
        dataset: the name of the dataset
        preprocessing: a list of preprocessing steps identified via the function name
        dataset_dir: a valid Path to dataset
        epochs:
        batch_size:
        train_split:
        val_split: on which split to evaluate on (one of: "val" / "test")
        use_tpu: if there is a TPU Device which I should use
        serving_input_shape: the shape of the input tensor as a comma seperated string
        serving_input_key: the type of input
        throttle_after: seconds after which I throttle while evaluating

    Returns:
        None

    """
    tf.logging.info("entered training + continuous evaluation branch.")
    train_input_fn = functools.partial(
        get_data,
        dataset=dataset,
        preprocessing=preprocessing,
        dataset_dir=dataset_dir,
        split_name=train_split,
        is_training=True,
        num_epochs=epochs,
        drop_remainder=True,
        dataset_parameter=FLAGS,
    )
    eval_input_fn = functools.partial(
        get_data,
        dataset=dataset,
        preprocessing=preprocessing,
        dataset_dir=dataset_dir,
        split_name=val_split,
        is_training=False,
        shuffle=False,
        num_epochs=1,
        drop_remainder=use_tpu,
        dataset_parameter=FLAGS,
    )
    num_train_samples = get_count(dataset, train_split)
    updates_per_epoch = num_train_samples // batch_size
    num_train_steps = int(epochs * updates_per_epoch)
    estimator._export_to_tpu = False
    best_exporter = BestCheckpointCopier(
        name="best",
        checkpoints_to_keep=2,
        score_metric="loss",
        compare_fn=lambda x, y: x.score < y.score,
        sort_key_fn=lambda x: x.score,
        sort_reverse=False,
    )
    final_exporter = tf.estimator.FinalExporter(
        name="final_exporter",
        serving_input_receiver_fn=serving_input_fn(
            serving_input_shape, serving_input_key
        ),
    )
    exporters = (best_exporter, final_exporter)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=num_train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, exporters=exporters, throttle_secs=throttle_after
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def evaluate(
        estimator,
        model_dir: Path,
        dataset,
        preprocessing,
        dataset_dir,
        eval_batch_size: int = None,
        val_split="val",
        use_tpu=False,
        FLAGS={},
):
    """I evaluate the performance of a given estimator on a dataset.

    Args:
        estimator: a tensorflow.estimator.Estimator object
        model_dir: a Path to the directory containing the model to be evaluated
        dataset: the name of the dataset
        preprocessing: a list of preprocessing steps identified via the function name
        dataset_dir: a valid Path to dataset
        eval_batch_size: number of samples per evaluation batch
        val_split: on which split to evaluate on (one of: "val" / "test")
        use_tpu: if there is a TPU Device which I should use

    Returns:
        see tensorflow.estimator.Estimator.evaluate

    """
    tf.logging.info("entered run eval branch.")
    data_fn = functools.partial(
        get_data,
        dataset=dataset,
        preprocessing=preprocessing,
        dataset_dir=dataset_dir,
        split_name=val_split,
        is_training=False,
        shuffle=False,
        num_epochs=1,
        drop_remainder=use_tpu,
        dataset_parameter=FLAGS,
    )
    # Contrary to what the documentation claims, the `train` and the
    # `evaluate` functions NEED to have `max_steps` and/or `steps` set and
    # cannot make use of the iterator's end-of-input exception, so we need
    # to do some math for that here.
    num_samples = get_count(dataset, val_split)
    num_steps = num_samples // eval_batch_size
    tf.logging.info("val_steps: %d", num_steps)
    for checkpoint in tf.contrib.training.checkpoints_iterator(
            estimator.model_dir, timeout=10 * 60
    ):

        estimator.evaluate(
            checkpoint_path=checkpoint, input_fn=data_fn, steps=num_steps
        )

        hub_exporter = hub.LatestModuleExporter("hub", serving_input_fn)
        hub_exporter.export(estimator, (model_dir / "export/hub"), checkpoint)

        if tf.gfile.Exists(str(model_dir / "TRAINING_IS_DONE")):
            break
    # Evaluates the latest checkpoint on validation set.
    return estimator.evaluate(input_fn=data_fn, steps=num_steps)


def train(
        estimator,
        dataset,
        preprocessing,
        dataset_dir,
        epochs: int,
        batch_size: int,
        train_split="train",
        FLAGS={},
):
    """I train a tensorflow estimator on the given dataset.

    Args:
        estimator: a tensorflow.estimator.Estimator object
        dataset: the name of the dataset
        preprocessing: a list of preprocessing steps identified via the function name
        dataset_dir: a valid Path to dataset
        epochs: number of epochs to be trained
        batch_size: number of samples per batch
        train_split: on which split to train (one of: "train" / "trainval")

    Returns:
        estimator (after training for chaining)

    """
    tf.logging.info("entered training branch.")
    train_data_fn = functools.partial(
        get_data,
        dataset=dataset,
        preprocessing=preprocessing,
        dataset_dir=dataset_dir,
        split_name=train_split,
        is_training=True,
        num_epochs=int(math.ceil(epochs)),
        drop_remainder=True,
        dataset_parameter=FLAGS,
    )
    # We compute the number of steps and make use of Estimator's max_steps
    # arguments instead of relying on the Dataset's iterator to run out after
    # a number of epochs so that we can use 'fractional' epochs, which are
    # used by regression tests. (And because TPUEstimator needs it anyways.)
    num_samples = get_count(dataset, train_split)
    # Depending on whether we drop the last batch each epoch or only at the
    # ver end, this should be ordered differently for rounding.
    updates_per_epoch = num_samples // batch_size
    num_steps = int(math.ceil(epochs * updates_per_epoch))
    tf.logging.info("train_steps: %d", num_steps)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs", sess.graph)

    return estimator.train(train_data_fn, steps=num_steps)


def serving_input_fn(serving_input_shape, serving_input_key):
    """A serving input fn."""
    input_shape = str2intlist(serving_input_shape)
    image_features = {
        serving_input_key: tf.placeholder(dtype=tf.float32, shape=input_shape)
    }
    return tf.estimator.export.ServingInputReceiver(
        features=image_features, receiver_tensors=image_features
    )


def get_dependend_flags():
    with open("self_supervised_3d_tasks/dependend_flags.json", "r") as f:
        return json.load(f)


def check_task_dependend_flags(flags):
    dependend_flags = get_dependend_flags()
    # TODO: rotate3d, crop_patches3d are possible preprocessing step

    # test dependend flags on task (e.g. jigsaw)
    task = flags["task"]
    for flag in dependend_flags["dependend_flags_of_tasks"][task]["required"]:
        if not flags[flag]:
            raise MissingFlagsError(task, flag)

    # test dependent flags on architecture (e.g. resnet50)
    architecture = flags["architecture"]
    for flag in dependend_flags["dependend_flags_of_architectures"][architecture][
        "required"
    ]:
        if not flags[flag]:
            raise MissingFlagsError(architecture, flag)


def main():
    pass


if __name__ == "__main__":
    logging.basicConfig(filename="train_and_eval.log", level=logging.INFO)
    logging.info("Started the script.")
    parser = argparse.ArgumentParser(description="Train and Evaluation Pipeline")

    # General run setup flags.
    parser.add_argument(
        "--workdir", type=Path, help="Where to store files.", required=True
    )

    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--use_tpu", type=bool, default=False, help="Whether running on TPU or not."
    )
    parser.add_argument("--run_eval", type=bool, default=False, help="Run eval mode")
    parser.add_argument(
        "--train_eval",
        type=bool,
        default=True,
        help="Run eval every once in a while during training mode",
    )

    parser.add_argument(
        "--tpu_worker_name", default="tpu_worker", help="Name of a TPU worker."
    )

    # More detailed experiment flags
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset to use, typically " "`imagenet`.",
        required=True,
    )

    parser.add_argument(
        "--dataset_dir", type=Path, help="Location of the dataset files.", required=True
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,  # TODO: fill in what is used when *no* path is selected
        help="Location of a pretrained checkpoint file to load weights from.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="Optional different batch-size"
        " evaluation, defaults to the same as `batch_size`.",
    )

    parser.add_argument(
        "--keep_checkpoint_every_n_hours",
        type=int,
        help="Keep one "
        "checkpoint every this many hours. Otherwise, only the "
        "last few ones are kept. Defaults to 4h.",
    )

    parser.add_argument("--random_seed", type=int, help="Seed to use. None is random.")

    parser.add_argument(
        "--save_checkpoints_secs",
        type=int,
        help="Every how many seconds "
        "to save a checkpoint. Defaults to 600 ie every 10mins.",
    )

    parser.add_argument(
        "--throttle_secs",
        type=int,
        help="Every how many seconds "
        "to run evaluation on val dataset. Defaults to 90 ie every 10mins.",
    )

    parser.add_argument(
        "--serving_input_key",
        type=str,
        default="image",
        help="The name of the input tensor "
        "in the generated hub module. Just leave it at default.",
    )

    parser.add_argument(
        "--serving_input_shape",
        type=str,
        default="None,None,None,3",
        help="The shape of the input tensor"
        " in the stored hub module. Can contain `None`.",
    )

    parser.add_argument(
        "--signature",
        type=str,
        help="The name of the tensor to use as "
        "representation for evaluation. Just leave to default.",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Which pretext-task to learn from. Can be "
        "one of `rotation`, `exemplar`, `jigsaw`, "
        "`relative_patch_location`, `linear_eval`, `supervised_classification`, "
        "`supervised_segmentation`.",
        required=True,
    )

    parser.add_argument(
        "--train_split",
        type=str,
        help="Which dataset split to train on. "
        "Should only be `train` (default) or `trainval`.",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        help="Which dataset split to eval on. "
        "Should only be `val` (default) or `test`.",
    )

    # Flags about the pretext tasks

    parser.add_argument(
        "--embed_dim",
        type=int,
        help="For most pretext tasks, which "
        "dimension the embedding/hidden vector should be. "
        "Defaults to 1000.",
    )

    parser.add_argument(
        "--margin",
        type=float,
        help="For the `exemplar` pretext task, "
        "how large the triplet loss margin should be.",
    )

    parser.add_argument(
        "--num_of_inception_patches",
        type=int,
        help="For the Exemplar "
        "pretext task, how many instances of an image to create.",
    )

    parser.add_argument(
        "--fast_mode",
        type=bool,
        default=True,
        help="For the Exemplar "
        "pretext task, whether to distort color in fast mode or not.",
    )

    parser.add_argument(
        "--patch_jitter",
        type=int,
        help="For patch-based methods, by how "
        "many pixels to jitter the patches. Defaults to 0.",
    )

    parser.add_argument(
        "--perm_subset_size",
        type=int,
        help="Subset of permutations to "
        "sample per example in the `jigsaw` pretext task. "
        "Defaults to 8.",
    )

    parser.add_argument(
        "--splits_per_side",
        type=int,
        help="For the `crop_patches` "
        "preprocessor, how many times to split a side. "
        "For example, 3 will result in 3x3=9 patches.",
    )

    # Flags for evaluation.
    parser.add_argument(
        "--eval_model",
        type=str,
        help="Whether to perform evaluation with a "
        "`linear` (default) model, or with an `mlp` model.",
    )

    parser.add_argument(
        "--hub_module",
        type=str,
        help="Folder where the hub module that " "should be evaluated is stored.",
    )

    parser.add_argument(
        "--pool_mode",
        type=str,
        help="When running evaluation on "
        "intermediate layers (not logits) of the network, it is "
        "commonplace to pool the features down to 9000. This "
        "decides the pooling method to be used: `adaptive_max` "
        "(default), `adaptive_avg`, `max`, or `avg`.",
    )

    parser.add_argument(
        "--combine_patches",
        type=str,
        help="When running evaluation on "
        "patch models, it is used to merge patch representations"
        "to the full image representation. The value should be set"
        "to `avg_pool`(default), or `concat`.",
    )

    # Flags about the model.
    parser.add_argument(
        "--architecture",
        type=str,
        help="Which basic network architecture to use. "
        "One of vgg19, resnet50, revnet50, unet_resnet50.",
    )
    # flags.mark_flag_as_required('architecture')  # Not required in eval mode.

    parser.add_argument(
        "--filters_factor",
        type=int,
        help="Widening factor for network "
        "filters. For ResNet, default = 4 = vanilla ResNet.",
    )

    parser.add_argument(
        "--last_relu",
        type=bool,
        default=True,
        help="Whether to include (default) the final "
        "ReLU layer in ResNet/RevNet models or not.",
    )

    parser.add_argument(
        "--resnet_mode", type=str, default="v2", help="Which ResNet to use, `v1` or `v2`."
    )

    # Flags about the optimization process.
    parser.add_argument(
        "--batch_size", type=int, help="The global batch-size to use.", required=True
    )

    parser.add_argument(
        "--decay_epochs",
        type=str,
        help="Optional list of epochs at which "
        "learning-rate decay should happen, such as `15,25`.",
    )

    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to run training.", default=5
    )

    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        help="Factor by which to decay the "
        "learning-rate at each decay step. Default 0.1.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="The base learning-rate to use for training.",
        required=True,
    )

    parser.add_argument(
        "--lr_scale_batch_size",
        type=float,
        help="The batch-size for which the "
        "base learning-rate `lr` is defined. For batch-sizes "
        "different from that, it is scaled linearly accordingly."
        "For example lr=0.1, batch_size=128, lr_scale_batch_size=32"
        ", then actual lr=0.025.",
        required=True,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        help="Which optimizer to use. " "Only `sgd` (default) or `adam` are supported.",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        help="Duration of the linear learning-"
        "rate warm-up (from 0 to actual). Defaults to 0.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Strength of weight-decay. Defaults to 1e-4, and may be set to 0.",
    )

    # Flags about pre-processing/data augmentation.
    parser.add_argument(
        "--crop_size",
        type=str,
        help="Size of the crop when using `crop` "
        "or `central_crop` preprocessing. Either a single "
        "integer like `32` or a pair like `32,24`.",
    )

    parser.add_argument(
        "--grayscale_probability",
        type=float,
        help="When using `to_gray` "
        "preprocessing, probability of actually doing it. Defaults "
        "to 1.0, i.e. deterministically grayscaling the input.",
    )

    parser.add_argument(
        "--preprocessing",
        type=lambda s: [item for item in s.split(",")],
        help="A comma-separated list of pre-processing steps to perform, see preprocess.py.",
    )
    # flags.mark_flag_as_required("preprocessing")  # TODO: necessary?

    parser.add_argument(
        "--randomize_resize_method",
        type=bool,
        help="Whether or not (default) "
        "to use a random interpolation method in the `resize` "
        "preprocessor.",
    )

    parser.add_argument(
        "--resize_size",
        type=str,
        help="For the `resize`, "
        "`inception_preprocess`, and "
        "`crop_inception_preprocess_patches` preprocessors, the "
        "size in pixels to which to resize the input. Can be a "
        "single number for square, or a pair as `128,64`.",
    )

    parser.add_argument(
        "--smaller_size",
        type=int,
        help="For the `resize_small` preprocessor"
        ", the desired size that the smaller side should have "
        "after resizing the image (keeping aspect ratio).",
    )

    flags = parser.parse_args()
    logging.info(flags)

    check_task_dependend_flags(vars(flags))
    train_and_eval(vars(flags))
    logging.info("I'm done with my work, ciao!")
