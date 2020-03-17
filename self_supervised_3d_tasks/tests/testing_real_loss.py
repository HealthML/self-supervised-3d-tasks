from self_supervised_3d_tasks.data.segmentation_task_loader import SegmentationGenerator3D

from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms.keras_test_algo import score_jaccard, score_dice
from self_supervised_3d_tasks.keras_algorithms.losses import weighted_sum_loss, weighted_dice_coefficient

import tensorflow.keras.backend as K
import numpy as np

import tensorflow as tf

def qwck(y_true, y_pred, num_classes=5):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred = tf.cast(y_pred, dtype=tf.int32)

    if y_true.shape.as_list() != y_pred.shape.as_list():
        raise ValueError(
            "Number of samples in y_true and y_pred are different")

    # compute the new values of the confusion matrix
    new_conf_mtx = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=num_classes,
        weights=None)

    nb_ratings = tf.shape(new_conf_mtx)[0]
    weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.int32)

    # 2. Create a weight matrix
    weight_mtx += tf.range(nb_ratings, dtype=tf.int32)
    weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

    weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)
    weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

    # 3. Get counts
    actual_ratings_hist = tf.reduce_sum(new_conf_mtx, axis=1)
    pred_ratings_hist = tf.reduce_sum(new_conf_mtx, axis=0)

    # 4. Get the outer product
    out_prod = pred_ratings_hist[..., None] * \
               actual_ratings_hist[None, ...]

    # 5. Normalize the confusion matrix and outer product
    conf_mtx = new_conf_mtx / tf.reduce_sum(new_conf_mtx)
    out_prod = out_prod / tf.reduce_sum(out_prod)

    conf_mtx = tf.cast(conf_mtx, dtype=tf.float32)
    out_prod = tf.cast(out_prod, dtype=tf.float32)

    # 6. Calculate Kappa score
    numerator = tf.reduce_sum(conf_mtx * weight_mtx)
    denominator = tf.reduce_sum(out_prod * weight_mtx)
    kp = 1 - (numerator / denominator)
    return kp

if __name__ == "__main__":
    y1 = np.array([0,0,1,0,0])
    y2 = np.array([0,1,0,0,0])

    y1 = tf.constant(y1)
    y2 = tf.constant(y2)

    print(qwck(y1,y2).eval(session=K.get_session()))

def test():
    x = np.ones((1,))
    x /= 0

    # print(x)
    # print(np.clip(float("nan"), K.epsilon(), 1 - K.epsilon()))

    # gl = tfa.losses.GIoULoss()
    # boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    # boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    # loss = gl(boxes1, boxes2)
    # print('Loss: ', loss.numpy())

    path = "/home/Shared.Workspace/data/pancreas/images_resized_128_labeled/train"
    gen = get_data_generators(path, SegmentationGenerator3D)

    y_batch = gen[0][1]
    print(y_batch)
    y_pred = np.zeros(y_batch.shape)
    y_pred[:, :, :, :, 0] = 1

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    loss = weighted_dice_coefficient(K.constant(y_batch), K.constant(y_pred)).numpy()

    print(loss)
    print(score_dice(y_batch, y_pred))

    # weights = [0.1, 100, 125]
    # loss_weighted = weighted_sum_loss(0.25, 0.75, weights)(y_batch, y_pred).eval(session=K.get_session())

