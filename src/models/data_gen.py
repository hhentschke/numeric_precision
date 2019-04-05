#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:40:41 2019

@author: hh
"""
import sys
from keras.preprocessing.image import ImageDataGenerator
# collection of metrics
sys.path.append("/mnt/hd_internal/hh/projects_DS/numeric_precision/")


def wrapper_flow_from_directory(generator):
    """
    Wrapper for a Keras ImageDataGenerator.flow_from_directory which returns the labels duplicated
     in a list (that is, [y, y]) for usage with models with two outputs. 
     Input must be an instance of ImageDataGenerator.flow_from_directory.
    """
    for x, y in generator:
        yield x, [y, y]
    

def get_ImageDataGenerator(img_dir, batch_size, doImageTransform=False):
    """
    Returns an instance of Keras's ImageDataGenerator.flow_from_directory which 
    has been enhanced to produce labels compatible with two output layers which 
    require the same labels.
    """
    if doImageTransform:
        datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(
        rescale=1./255)
    
    generator = datagen.flow_from_directory(
        img_dir,
        target_size=(150, 150),
        batch_size=batch_size, 
        class_mode="binary"
    )
    return wrapper_flow_from_directory(generator)