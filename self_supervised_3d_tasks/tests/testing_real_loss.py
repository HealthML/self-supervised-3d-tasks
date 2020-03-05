from self_supervised_3d_tasks.data.segmentation_task_loader import SegmentationGenerator3D

from self_supervised_3d_tasks.data.make_data_generator import get_data_generators
from self_supervised_3d_tasks.keras_algorithms.losses import weighted_sum_loss

import tensorflow.keras.backend as K
import numpy as np

if __name__ == "__main__":
    x = np.ones((1,))
    x /= 0

    print(x)
    print(np.clip(float("nan"), K.epsilon(), 1 - K.epsilon()))

    # gl = tfa.losses.GIoULoss()
    # boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    # boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    # loss = gl(boxes1, boxes2)
    # print('Loss: ', loss.numpy())

    path = "/home/Shared.Workspace/data/pancreas/images_resized_128_labeled/train"
    gen = get_data_generators(path, SegmentationGenerator3D)

    y_batch = gen[0][1]
    y_pred = np.zeros(y_batch.shape)
    y_pred[:, :, :, :, 1] = 1

    weights = [0.1, 100, 125]
    loss_weighted = weighted_sum_loss(0.25, 0.75, weights)(y_batch, y_pred).eval(session=K.get_session())

