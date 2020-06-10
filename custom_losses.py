# -*- coding: utf-8 -*-
'''
Capsules for Object Segmentation (SegCaps)
Original Paper: https://arxiv.org/abs/1804.04241
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of custom loss functions not present in the default Keras.
'''

import tensorflow as tf
from keras import backend as K
import numpy as np
import itertools

def dice_metric(y_true, y_pred, axis=[1, 2], smooth=1e-7):
    #inse = K.sum(y_pred * y_true, axis=axis)
    #l = K.sum(y_pred, axis=axis)
    #r = K.sum(y_true, axis=axis)
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    dice_avg = 0
    for cl in range(1,4): #ignore background
        pred_cl = K.equal(y_pred, cl)
        true_cl = K.equal(y_true, cl)
        pred_cl = K.cast(pred_cl, dtype='float16')
        true_cl = K.cast(true_cl, dtype='float16')
        inse = K.sum(pred_cl*true_cl, axis=[1, 2])
        l = K.sum(pred_cl, axis=[1, 2])
        r = K.sum(true_cl, axis=[1, 2])
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = K.mean(dice, axis=0)
        dice_avg = dice_avg + dice

    dice_avg = dice_avg / 3

   # dice = (2. * inse + smooth) / (l + r + smooth)
   # dice = K.mean(dice, axis=-1)
   # dice = K.mean(dice, axis=0)
    return dice_avg



def dice_soft(y_true, y_pred, loss_type='sorensen', axis=(0, 1, 2), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    y_pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_true : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both y_pred and y_true are empty, it makes sure dice is 1.
        If either y_pred or y_true are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """

    # if not from_logits:
    #     # transform back to logits
    #     _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
    #     y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    #     y_pred = tf.log(y_pred / (1 - y_pred))
    # n_labels = y_true.get_shape()[3]
    n_labels = tf.shape(y_true)[3]
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), n_labels)

    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    # if loss_type == 'jaccard':
    #     l = tf.reduce_sum(y_pred * y_pred, axis=axis)
    #     r = tf.reduce_sum(y_true * y_true, axis=axis)
    # elif loss_type == 'sorensen':
    #     l = tf.reduce_sum(y_pred, axis=axis)
    #     r = tf.reduce_sum(y_true, axis=axis)
    # else:
    #     raise Exception("Unknow loss_type")
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def dice_hard(y_true, y_pred, threshold=0.5, axis=(0, 1, 2), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.

    Parameters
    -----------
    y_pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_true : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def weighted_binary_crossentropy_loss(pos_weight):
    # pos_weight: A coefficient to use on the positive examples.
    def weighted_binary_crossentropy(target, output, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))

        return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                        logits=output,
                                                        pos_weight=pos_weight)

    return weighted_binary_crossentropy


def margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0):
    '''
    Args:
        margin: scalar, the margin after subtracting 0.5 from raw_logits.
        downweight: scalar, the factor for negative cost.
    '''

    def _margin_loss(labels, raw_logits):
        """Penalizes deviations from margin for each logit.

        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.

        Args:
        labels: tensor, one hot encoding of ground truth.
        raw_logits: tensor, model predictions in range [0, 1]


        Returns:
        A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = pos_weight * labels * tf.cast(tf.less(logits, margin),
                                                      tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(
            tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    return _margin_loss


class WeightedCategoricalCrossEntropy(object):

    def __init__(self, weights):
        nb_cl = len(weights)
        self.weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'w_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.w_categorical_crossentropy(y_true, y_pred)

    def w_categorical_crossentropy(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_true[..., c_t], K.floatx())
            final_mask += w * y_p * y_t
        return K.categorical_crossentropy(y_true, y_pred) * final_mask


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def weighted_dice(y_true, y_pred, weights, smooth=1e-7):
    if not weights:
        w = K.sum(y_true, axis=(0, 1, 2))
        w = K.sum(w) / (w + K.constant(1, dtype=tf.float32))
        w = w / K.max(w)
    else:
        w = K.constant(weights)
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), 4)
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, axis=(0, 1, 2))
    numerator = K.sum(numerator)

    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, axis=(0, 1, 2))
    denominator = K.sum(denominator)

    return (2. * numerator + smooth) / (denominator + smooth)


def weighted_dice_coef(weights=()):
    def coef(y_true, y_pred, from_logits=False):
        return weighted_dice(y_true, y_pred, weights)

    return coef


def weighted_dice_loss(weights=()):
    def loss(y_true, y_pred, from_logits=False):
        return 1 - weighted_dice(y_true, y_pred, weights)

    return loss


def weighted_mse_loss(weight=-1):
    def weighted_mse(y_true, y_pred, smooth=1e-7):
        if weight == -1:
            n_el = K.constant(256 * 256, dtype='int64')
            count_positive = tf.math.count_nonzero(y_true)
            count_zero = n_el - count_positive
            w_zero = K.switch(
                K.greater(count_zero, count_positive),
                count_positive / count_zero + K.constant(smooth, dtype='float64'),
                K.constant(1, dtype='float64')
            )
            w_positive = K.switch(
                K.greater(count_positive, count_zero),
                count_zero / count_positive + K.constant(smooth, dtype='float64'),
                K.constant(1, dtype='float64')
            )
            w = tf.where(K.equal(y_true, K.constant(0)), w_zero, w_positive)
        else:
            w = tf.where(K.equal(y_true, K.constant(0)), K.constant(1, dtype='float32'), K.constant(weight))

        return K.mean(w * K.square(y_true - y_pred))

    return weighted_mse


def spread_loss(m_low=0.2, m_high=0.9, epochs=100, epoch_step=100):
    def loss_fun(labels, logits):
        n_labels = tf.shape(labels)[3]
        m = m_low + (m_high - m_low) * tf.minimum(tf.cast(epoch_step / epochs, dtype=tf.float32),
                                                  tf.cast(1, dtype=tf.float32))
        # n_labels = labels.get_shape()[3]
        labels = tf.transpose(labels, (3, 0, 1, 2))
        logits = tf.transpose(logits, (3, 0, 1, 2))
        labels = tf.reshape(labels, [n_labels, -1])
        logits = tf.reshape(logits, [n_labels, -1])

        true_class_logits = tf.reduce_max(labels * logits, axis=0)
        margin_loss_pixel_class = tf.square(tf.nn.relu((m - true_class_logits + logits) * (1 - labels)))

        loss = tf.reduce_mean(tf.reduce_sum(margin_loss_pixel_class, axis=0))

        return loss

    return loss_fun


def weighted_spread_loss(weights=None, m_low=0.2, m_high=0.9, epochs=50, epoch_step=50):
    def loss_fun(labels, logits):
        # w_l = np.array([0.00705479, 0.03312549, 0.02664785, 0.4437354, 0.44254721, 0.04688926]) * 6
        n_labels = tf.shape(labels)[3]
        m = m_low + (m_high - m_low) * tf.minimum(tf.cast(epoch_step / epochs, dtype=tf.float32),
                                                  tf.cast(1, dtype=tf.float32))
        # n_labels = labels.get_shape()[3]
        labels = tf.transpose(labels, (3, 0, 1, 2))
        logits = tf.transpose(logits, (3, 0, 1, 2))
        labels = tf.reshape(labels, [n_labels, -1])
        logits = tf.reshape(logits, [n_labels, -1])

        true_class_logits = tf.reduce_max(labels * logits, axis=0)
        margin_loss_pixel_class = tf.square(tf.nn.relu((m - true_class_logits + logits) * (1 - labels)))

        margin_loss_pixel_class = tf.transpose(margin_loss_pixel_class, (1, 0))
        if weights is not None:
            margin_loss_pixel_class = weights * margin_loss_pixel_class

        # loss = tf.reduce_mean(tf.reduce_sum(margin_loss_pixel_class, axis=1))
        loss = tf.reduce_sum(tf.reduce_mean(margin_loss_pixel_class, axis=0))

        return loss

    return loss_fun


# def spread_loss(m_low=0.2, m_high=0.9, epochs=20, epoch_step=20):
#     return weighted_spread_loss(m_low=m_low, m_high=m_high, epochs=epochs, epoch_step=epoch_step)

