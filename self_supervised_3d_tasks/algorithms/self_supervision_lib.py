"""Generates training data with self_supervised supervision.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inspect

from . import exemplar, supervised_classification, jigsaw, rotation
from . import relative_patch_location, supervised_segmentation

from ..errors import MissingFlagsError


def check_for_missing_arguments(model_fn, kwargs):
    model_fn_parameters = inspect.signature(model_fn).parameters.values()
    missing_flags = []
    for param in model_fn_parameters:
        if isinstance(param.default, inspect._empty.__class__):
            # the argument has no default value --> required parameter --> check if given!
            if not param.name in kwargs:
                missing_flags.append(param.name)
    return missing_flags


def collect_model_kwargs(model_fn, kwargs):
    model_fn_parameters = inspect.signature(model_fn).parameters.values()
    kwargs_collected = {}
    for param in model_fn_parameters:
        if param.name in kwargs:
            kwargs_collected[param.name] = kwargs[param.name]
    return kwargs_collected


def get_self_supervision_model(self_supervision, model_kwargs={}):
    """Gets self_supervised supervised training data and labels."""

    mapping = {
        "supervised_classification": supervised_classification.model_fn,
        "supervised_segmentation": supervised_segmentation.model_fn,
        "rotation": rotation.model_fn,
        "jigsaw": jigsaw.model_fn,
        "relative_patch_location": relative_patch_location.model_fn,
        "exemplar": exemplar.model_fn,
    }

    model_fn = mapping.get(self_supervision)
    if model_fn is None:
        raise ValueError("Unknown self_supervised-supervision: %s" % self_supervision)

    missing_flags = check_for_missing_arguments(model_fn, model_kwargs)
    if missing_flags:
        raise MissingFlagsError(self_supervision, missing_flags)


    def _model_fn(features, labels, mode, params):
        """Returns the EstimatorSpec to run the model.

        Args:
          features: Dict of inputs ("image" being the image).
          labels: unused but required by Estimator API.
          mode: model's mode: training, eval or prediction
          params: required by Estimator API, contains TPU local `batch_size`.

        Returns:
          EstimatorSpec

        Raises:
          ValueError when the algorithms is unknown.
        """
        del labels, params  # unused
        tf.logging.info("Calling model_fn in mode %s with data:", mode)
        tf.logging.info(features)
        tf.logging.info("Parameters: ", **model_kwargs)

        return model_fn(features, mode, **collect_model_kwargs(model_fn, model_kwargs))

    return _model_fn
