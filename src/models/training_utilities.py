#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:48:29 2019

@author: hh
"""

def lr_schedule(epoch_ix, lr, mode='step_decay'):
    """
    Defines learning rate schedule, to be used in keras.Callbacks.LearningRateScheduler.
    Input
        epoch_ix - index of epoch
        lr - current learning rate
        mode - 'step_decay'
    Returns
        learning rate as a function of lr and epoch_ix
    -- Work in progress --
    """
    assert(mode in ["step_decay"])
    
    if mode is "step_decay":
        # step size (number of episodes)
        step_size = 10
        # factor by which to multiply learning rate after each step
        decay_factor = 0.75
        if ((epoch_ix > 0) and ((epoch_ix % step_size) == 0)):
            lr = lr * decay_factor
        
    return lr

# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1