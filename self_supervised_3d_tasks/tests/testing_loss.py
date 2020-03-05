import numpy as np
import os

import tensorflow as tf
from sklearn.metrics import jaccard_score

from self_supervised_3d_tasks.keras_algorithms.keras_test_algo import score_jaccard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow_core.python.keras.metrics import CategoricalCrossentropy

from self_supervised_3d_tasks.data.segmentation_task_loader import SegmentationGenerator3D

from self_supervised_3d_tasks.data.make_data_generator import get_data_generators

EPSILON = 1e-07


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array, list or tuple of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')

    @url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    @author: wassname
    """
    if isinstance(weights, list) or isinstance(weights, tuple):
        weights = np.array(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= np.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        # calc
        loss = y_true * np.log(y_pred) * weights
        loss = -np.sum(loss, -1)
        loss = np.mean(loss)

        return loss

    return loss


def jaccard_distance_XX(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    This loss expects one hot encoded labels!
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    @url:https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    @author: wassname, ahundt
    """
    intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
    sum_ = np.sum(np.abs(y_true) + np.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth + EPSILON)
    return (1 - jac) * smooth


def jaccard_distance(y_true, y_pred, smooth=25):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = np.sum(np.abs(y_true * y_pred), axis=tuple(range(y_pred.ndim - 1)))
    sum_ = np.sum(np.abs(y_true) + np.abs(y_pred), axis=tuple(range(y_pred.ndim - 1)))

    jac = (intersection + smooth) / (sum_ - intersection + smooth + EPSILON)
    jd = np.mean(jac)

    return (1 - jd) * smooth



def weighted_sum_loss(alpha=1, beta=1, weights=(1, 5, 10)):
    w_cross_entropy = weighted_categorical_crossentropy(weights)

    def loss(y_true, y_pred):
        gdl = alpha * jaccard_distance(y_true=y_true, y_pred=y_pred)
        wce = beta * w_cross_entropy(y_true=y_true, y_pred=y_pred)
        return gdl + wce

    return loss


if __name__ == "__main__":
    path = "/home/Shared.Workspace/data/pancreas/images_resized_128_labeled/train"
    gen = get_data_generators(path, SegmentationGenerator3D)

    loss = weighted_categorical_crossentropy([0.1,100,125])

    y_batch = gen[0][1]
    y_pred = np.zeros(y_batch.shape)
    #print(np.sum(np.abs(y_pred - y_batch)))
    #print(y_pred.shape)

    y_pred[:, :, :, :, 1] = 1
    #print(np.sum(np.abs(y_pred - y_batch)))

    xx = loss(y_batch, y_pred)
    xxxxx = weighted_sum_loss()(y_batch, y_pred)

    #print(xx.shape)
    #print(xx.max())
    #print(xx.min())

    print("WEIGHTED CE")
    print(np.average(xx))

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    m = CategoricalCrossentropy()
    m.update_state(y_batch, y_pred)

    print("CAT CE")
    print(m.result().numpy())

    print("JACCARD")
    print(np.average(jaccard_distance(y_batch, y_pred)))

    print("OUR SCORE")
    print(score_jaccard(y_batch, y_pred))