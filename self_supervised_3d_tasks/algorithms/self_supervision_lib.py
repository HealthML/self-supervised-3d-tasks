"""Generates training data with self_supervised supervision.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from self_supervised_3d_tasks.algorithms import contrastive_predictive_coding

from ..dependend_flags.flag_utils import (
    check_for_missing_arguments,
    collect_model_kwargs,
)
from . import exemplar, supervised_classification, jigsaw, rotation
from . import relative_patch_location, supervised_segmentation

from ..errors import MissingFlagsError


def get_self_supervision_model(self_supervision, model_kwargs={}):
    """Gets self_supervised supervised training data and labels."""

    mapping = {
        "supervised_classification": supervised_classification.model_fn,
        "supervised_segmentation": supervised_segmentation.model_fn,
        "rotation": rotation.model_fn,
        "jigsaw": jigsaw.model_fn,
        "relative_patch_location": relative_patch_location.model_fn,
        "exemplar": exemplar.model_fn,
        "cpc": contrastive_predictive_coding.model_fn,
    }

    model_fn = mapping.get(self_supervision)
    if model_fn is None:
        raise ValueError("Unknown self_supervised-supervision: %s" % self_supervision)

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
        tf.logging.info("Parameters: ", *model_kwargs)

        model_fn_mapped = functools.partial(
            model_fn,
            data=features,
            mode=mode,
            net_params=model_kwargs)

        missing_flags = check_for_missing_arguments(model_fn_mapped, model_kwargs)
        if missing_flags:
            raise MissingFlagsError(self_supervision, missing_flags)

        return model_fn_mapped(**collect_model_kwargs(model_fn, model_kwargs))

    return _model_fn
