import json
import os
from pathlib import Path
import tensorflow as tf
import numpy as np
from self_supervised_3d_tasks.algorithms import patch3d_utils
from self_supervised_3d_tasks.algorithms.patch3d_utils import apply_model
from self_supervised_3d_tasks.train_and_eval import train_and_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

with tf.Session() as sess:
    path = '/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/relative_patch_location/'

    with open(Path(__file__).parent.parent / "config/relative_patch_location/ukb3d.json", "r") as f:
        args = json.load(f)

    X = tf.placeholder(
        shape=(54, 32, 32, 32, 2), dtype=tf.float32)

    def input_fn(params):
        return {"image":X}

        #
        # return tf.Variable(tf.zeros((54, 32, 32, 32, 2)))

    perms, num_classes = patch3d_utils.generate_patch_locations()

    with tf.variable_scope("module"):
        out = apply_model(lambda: X,
                          False,
                          num_classes,
                          perms,
                          2,
                          "unet_resnet3d_class",
                          "relative_patch_location",
                          net_params={"task": 0, "architecture": 0,"filters_factor": 2})

    saver = tf.train.Saver()
    saver.restore(sess, path + 'model.ckpt-0')

    # new_saver = tf.train.import_meta_graph(path + 'model.ckpt-0.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint(path))

    result = sess.run(out, feed_dict={X: np.zeros(shape=(54, 32, 32, 32, 2))})
    print(result)


def new():
    perms, num_classes = patch3d_utils.generate_patch_locations()

    with tf.variable_scope("module"):
        out = apply_model(input_fn,
                          False,
                          num_classes,
                          perms,
                          2,
                          "unet_resnet3d_class",
                          "relative_patch_location",
                          net_params={"task": 0, "architecture": 0})

    saver = tf.train.Saver()
    saver.restore(sess, path + 'model.ckpt-0')

    # new_saver = tf.train.import_meta_graph(path + 'model.ckpt-0.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint(path))

    result = sess.run(out)
    print(result)


def old():
    args["input_fn"] = input_fn
    args["predict"] = True

    result = train_and_eval(args)

    for x in result:
        print(x)

    # new_saver = tf.train.import_meta_graph(path+'model.ckpt-810612.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint(path))
