#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:48:29 2019

@author: hh
"""

def step_decay(epoch_ix, lr, step_size=10, decay_factor=0.5):
    """
    Defines 'step_decay'-type learning rate schedule, to be used in 
    keras.Callbacks.LearningRateScheduler.
    Input
        epoch_ix: index of epoch
        lr: current learning rate
        step_size: positive interger, number of episodes after which learning rate will adapt
        decay_factor: factor by which lr will be multiplied
    Returns
        learning rate as a function of lr and epoch_ix
    """
    if ((epoch_ix > 0) and ((epoch_ix % step_size) == 0)):
        lr = lr * decay_factor
    return lr

def lr_schedule_selector(mode="constant"):
    mode_dict = {
            "constant": lambda epoch_ix, lr: lr,
            "step_decay": step_decay}
    return mode_dict[mode]
    
# for further inspiration, look here:
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1