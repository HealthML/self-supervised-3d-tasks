"""Generates training data with self_supervised supervision.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import (
    exemplar,
    supervised_classification,
    jigsaw,
    rotation,
)
from . import (
    relative_patch_location,
    supervised_segmentation,
)


def get_self_supervision_model(self_supervision, model_args=[], model_kwargs={}):
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

        dict_model = [model_kwargs["architecture"]]

        dict_params = {
            "rotate3d" : model_kwargs["rotate3d"],
            "serving_input_shape" : model_kwargs["serving_input_shape"],
            "net_params" : model_kwargs
        }

        return model_fn(features, mode, *dict_model, **dict_params)

    return _model_fn
