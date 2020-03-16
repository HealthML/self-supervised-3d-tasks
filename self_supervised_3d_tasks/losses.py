import numpy as np
from tensorflow.keras import backend as K


def triplet_loss(y_true, y_pred, _alpha=1.0):
    """
    This function returns the calculated triplet loss for y_pred
    :param y_true: not needed
    :param y_pred: predictions of the model
    :param embedding_size: length of embedding
    :param _alpha: defines the shift of the loss
    :return: calculated loss
    """

    positive_distance = K.mean(
        K.square(y_pred[:, 0] - y_pred[:, 1]), axis=-1
    )
    negative_distance = K.mean(
        K.square(y_pred[:, 0] - y_pred[:, 2]), axis=-1
    )
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + _alpha))


def weighted_categorical_crossentropy(weights, debug=False):
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
    weights = K.variable(weights)

    def wcc_loss(y_true, y_pred):
        # scale predictions so that the class probs of each sample sum to 1
        ya = y_pred / (K.sum(y_pred, axis=-1, keepdims=True) + K.epsilon())
        # clip to prevent NaN's and Inf's
        yb = K.clip(ya, K.epsilon(), 1 - K.epsilon())
        # calc
        loss_a = y_true * K.log(yb) * weights
        loss_b = -K.sum(loss_a, -1)
        loss_c = K.mean(loss_b)

        return loss_c

    return wcc_loss


def jaccard_distance(y_true, y_pred, smooth=5):
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
    intersection = K.sum(K.abs(y_true * y_pred), axis=tuple(range(y_pred.shape.rank - 1)))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=tuple(range(y_pred.shape.rank - 1)))

    jac = (intersection + smooth) / (sum_ - intersection + smooth + K.epsilon())
    jd = K.mean(jac)

    return (1 - jd) * smooth

def weighted_dice_coefficient_per_class(y_true, y_pred, class_to_predict=0, smooth=0.00001):
    """
    Weighted dice coefficient.
    :param smooth:
    :param y_true:
    :param y_pred:
    :param class_to_predict:
    :return:
    """
    axis = tuple(range(y_pred.shape.rank - 1))

    return (2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))[class_to_predict]


def weighted_dice_coefficient(y_true, y_pred, smooth=0.00001):
    """
    Weighted dice coefficient.
    :param smooth:
    :param y_true:
    :param y_pred:
    :return:
    """
    axis = tuple(range(y_pred.shape.rank - 1))

    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def weighted_sum_loss(alpha=1, beta=1, weights=(1, 5, 10)):
    w_cross_entropy = weighted_categorical_crossentropy(weights)

    def loss(y_true, y_pred):
        gdl = alpha * jaccard_distance(y_true=y_true, y_pred=y_pred)
        wce = beta * w_cross_entropy(y_true=y_true, y_pred=y_pred)
        return gdl + wce

    return loss
