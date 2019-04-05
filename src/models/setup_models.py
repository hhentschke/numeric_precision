#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code file for network models. Each function should return not only the model
itself but also a name of the model composed of the crucial parameter choices
(for subsequent saving).
"""
from keras import models, layers
from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3


def convnet_01(input_shape):
    """
    Simple convolutional network for binary classification task with dual
    outputs, one of them with no activation/normalization, the other with
    sigmoid activation.
    Input args
        input_shape: shape of input (rows, cols, color channels of images)
    Output:
        model, model name
    """
    nm = ""
    model_name = "convnet01_{}".format(nm)
    ki = 'he_normal'

    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer=ki)(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer=ki)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer=ki)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer=ki)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    output_raw = layers.Dense(1, activation=None, name="raw")(x)
    output_sigmoid = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)

    model = models.Model(input, [output_raw, output_sigmoid])
    return model, model_name


def flex_base(input_shape, base_name, weights=None):
    """
    Network for binary classification task with specific base and dense layers on top.
    Can be invoked with or without pretrained weights. When invoked with pretrained 
    weights, the base is frozen. Dual outputs, one with no activation/normalization,
    the other with sigmoid activation.
    Input args
        input_shape: shape of input (rows, cols, color channels of images)
        base_name: name of network base ('VGG16' or 'InceptionV3')
        weights: weights of network base, 'imagenet' or None
    Output:
        model, model name
    """
    if weights is None:
        nm = ""
    else:
        nm = weights + "_"

    if base_name == "VGG16":
        model_name = "VGG16_{}".format(nm)
        model_base = VGG16(weights=weights,
                           include_top=False,
                           input_shape=input_shape)
    elif base_name == "InceptionV3":
        model_name = "InceptionV3_{}".format(nm)
        model_base = InceptionV3(weights=weights,
                                 include_top=False,
                                 pooling=None,
                                 input_shape=input_shape)
    else:
        raise Exception("bad base_name")

    # if weights are not None we want to freeze the convolutional layers
    if weights is not None:
        model_base.trainable = False
    input = layers.Input(shape=input_shape)
    x = model_base(input)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    output_raw = layers.Dense(1, activation=None, name="raw")(x)
    output_sigmoid = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)

    model = models.Model(input, [output_raw, output_sigmoid])
    return model, model_name
