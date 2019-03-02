#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:25:54 2019
A collection of deep learning 'metrics' for Keras with tensorflow backend.
@author: hh
"""

import keras.backend as K
import tensorflow as tf

def contorted_binary_crossentropy(target, output, from_raw=True):
    """
    Keras's binary crossentropy, slightly modified and with some seemingly 
    stupid code on top for illustration purposes. In contrast to the original 
    binary_crossentropy, this function expects raw/logits input.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_raw: Whether `output` is expected to be a raw tensor.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects raw values, Keras expects probabilities.

    # first, convert raw values and reset flag
    if from_raw:
        output = sigmoids(target, output)
        from_raw=False

    if not from_raw:
        # transform back to raw
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)


def bce_from_raw(y_true, y):
    """
    Binary crossentropy (BCE) computed from raw values. Essentially,
    a wrapper for tf.nn.sigmoid_cross_entropy_with_logits.
    Input arguments
        y_true: labels, a tensor
        y: *raw* output of last network layer; tensor of same size as y_true.
    Output
        BCE, a tensor of the same size as y_true and y
    """
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y)


def raw_values(y_true, y):
    """
    Returns input value y. Used for numerical diagnostics.
    """
    return y


def sigmoids(y_true, y):
    """
    Returns sigmoid of input value y. Used for numerical diagnostics.
    """
    one = tf.convert_to_tensor(1.0, tf.float32)
    return one / (one + tf.exp(-y))


def binary_accuracy_from_raw(y_true, y, threshold=0.0):
    """
    Accuracy for binary classification computed from raw values.
    Essentially, a wrapper for tf.keras.metrics.binary_accuracy
    with changed threshold.
    Input arguments
        y_true: labels, a tensor
        y: *raw* output of last network layer; tensor of same size as y_true.
        threshold: threshold
    Output
        Accuracy
    """
    return tf.keras.metrics.binary_accuracy(y_true, y, threshold=threshold)