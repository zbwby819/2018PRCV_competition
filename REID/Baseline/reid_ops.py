import numpy as np
import tensorflow as tf
import numpy.linalg as la

from Baseline.tf_sort_ops import argsort
from keras import backend as K
from Baseline.aug import aug_nhw3


"================================"


def triplet_loss(y_true, y_pred, batch_num):
    y_pred = K.l2_normalize(y_pred, axis=1)
    batch = batch_num
    ref1 = y_pred[0:batch, :]
    pos1 = y_pred[batch:batch + batch, :]
    neg1 = y_pred[batch + batch:3 * batch, :]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    a1 = 0.6
    d1 = K.maximum(0.0, dis_pos - dis_neg + a1)
    return K.mean(d1)


def triplet_hard_loss(y_true, y_pred, SN, PN, a1=0.6, msml=False):
    feat_num = SN * PN  # images num
    # y_pred = K.l2_normalize(y_pred, axis=1)
    feat1 = K.tile(K.expand_dims(y_pred, axis=0), [feat_num, 1, 1])
    feat2 = K.tile(K.expand_dims(y_pred, axis=1), [1, feat_num, 1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),
                    axis=2) + K.epsilon()  # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN, 0:SN]
    negetive = dis_mat[0:SN, SN:]
    for i in range(1, PN):
        positive = tf.concat(
            [positive, dis_mat[i * SN:(i + 1) * SN, i * SN:(i + 1) * SN]],
            axis=0)
        if i != PN - 1:
            negs = tf.concat([dis_mat[i * SN:(i + 1) * SN, 0:i * SN],
                              dis_mat[i * SN:(i + 1) * SN, (i + 1) * SN:]],
                             axis=1)
        else:
            negs = tf.concat(dis_mat[i * SN:(i + 1) * SN, 0:i * SN], axis=0)
        negetive = tf.concat([negetive, negs], axis=0)

    if msml:
        positive = K.max(positive)
        negetive = K.min(negetive)
    else:
        positive = K.max(positive, axis=1)
        negetive = K.min(negetive, axis=1)
    loss = K.mean(K.maximum(0.0, positive - negetive + a1))
    return loss


def softmargin_triplet_hard_loss(y_true, y_pred, SN, PN, lamd=0.01, msml=False):
    feat_num = SN * PN  # images num
    # y_pred = K.l2_normalize(y_pred, axis=1)
    feat1 = K.tile(K.expand_dims(y_pred, axis=0), [feat_num, 1, 1])
    feat2 = K.tile(K.expand_dims(y_pred, axis=1), [1, feat_num, 1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),
                    axis=2) + K.epsilon()  # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)

    # # ADD
    # dis_mat = K.l2_normalize(dis_mat, axis=1)

    positive = dis_mat[0:SN, 0:SN]
    negetive = dis_mat[0:SN, SN:]
    for i in range(1, PN):
        positive = tf.concat(
            [positive, dis_mat[i * SN:(i + 1) * SN, i * SN:(i + 1) * SN]],
            axis=0)
        if i != PN - 1:
            negs = tf.concat([dis_mat[i * SN:(i + 1) * SN, 0:i * SN],
                              dis_mat[i * SN:(i + 1) * SN, (i + 1) * SN:]],
                             axis=1)
        else:
            negs = tf.concat(dis_mat[i * SN:(i + 1) * SN, 0:i * SN], axis=0)
        negetive = tf.concat([negetive, negs], axis=0)
    if msml:
        positive = K.max(positive)
        negetive = K.min(negetive)
    else:
        positive = K.max(positive, axis=1)
        negetive = K.min(negetive, axis=1)
    # loss = K.mean(K.log(1 + K.exp(positive - negetive))) + lamd * K.mean(
    #     K.square(y_pred))
    loss = K.mean(K.log(1 + K.exp(positive - negetive)))
    # loss = K.mean(K.log(positive) - K.log(negetive))
    return loss


def evaluation_loss(y_true, y_pred, SN, PN):
    """
    calculate map
    :param y_true:
    :param y_pred:
    :param SN:
    :param PN:
    :return:
    """
    feat_num = SN * PN  # images num
    # y_pred = K.l2_normalize(y_pred, axis=1)
    feat1 = K.tile(K.expand_dims(y_pred, axis=0), [feat_num, 1, 1])
    feat2 = K.tile(K.expand_dims(y_pred, axis=1), [1, feat_num, 1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),
                    axis=2) + K.epsilon()  # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)

    positive = dis_mat[0:SN, 0:SN]
    negetive = dis_mat[0:SN, SN:]
    for i in range(1, PN):
        positive = tf.concat(
            [positive, dis_mat[i * SN:(i + 1) * SN, i * SN:(i + 1) * SN]],
            axis=0)
        if i != PN - 1:
            negs = tf.concat([dis_mat[i * SN:(i + 1) * SN, 0:i * SN],
                              dis_mat[i * SN:(i + 1) * SN, (i + 1) * SN:]],
                             axis=1)
        else:
            negs = tf.concat(dis_mat[i * SN:(i + 1) * SN, 0:i * SN], axis=0)
        negetive = tf.concat([negetive, negs], axis=0)

    conbined_mat = tf.concat([positive, negetive], axis=1)

    sort_mat = argsort(conbined_mat, axis=-1)

    tag_mat = np.zeros((SN * PN, SN * PN))
    tag_mat[:, 0:SN] = 1
    tag_mat = K.variable(tag_mat, dtype=tf.float32)

    ap = tf.Variable(0., dtype=tf.float32)

    for i in range(SN * PN):
        origin_sorted_tag = K.gather(tag_mat[i, :], sort_mat[i, :])
        accum_sorted_tag = K.cumsum(origin_sorted_tag[1:])
        accum_sorted_tag = accum_sorted_tag * origin_sorted_tag[1:]

        ap = ap + K.sum(
            tf.div(accum_sorted_tag,
                   tf.range(start=1, limit=SN * PN, dtype=tf.float32))) / (
                         SN - 1)

    map = ap / (SN * PN)

    return map


def tf_debug_print(tensor):
    with tf.Session():
        print(tensor.eval())


"================================"
if __name__ == '__main__':
    with tf.Session() as sess:
        tag_mat = np.zeros((4, 4))
        tag_mat[:, 0:2] = 1
        tag_mat = tf.Variable(tag_mat, dtype=tf.float32)

        a = tf.Variable(
            np.array([[2, 3, 1, 4], [4, 2, 1, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
        sort_mat = argsort(a, axis=-1)

        ap = tf.Variable(0., dtype=tf.float32)

        for i in range(4):
            origin_sorted_tag = K.gather(tag_mat[i, :], sort_mat[i, :])
            accum_sorted_tag = K.cumsum(origin_sorted_tag[1:])
            accum_sorted_tag = accum_sorted_tag * origin_sorted_tag[1:]

            ap = ap + K.sum(
                tf.div(accum_sorted_tag,
                       tf.range(start=1, limit=4, dtype=tf.float32)))

        map = ap / 12

        sess.run(tf.global_variables_initializer())
        sess.run(sort_mat)
        # sess.run(origin_sorted_tag)
        # sess.run(accum_sorted_tag)
        sess.run(map)

        print(sort_mat.eval())
        print(tag_mat.eval())
        # print(origin_sorted_tag.eval())
        # print(accum_sorted_tag.eval())
        print(map.eval())
