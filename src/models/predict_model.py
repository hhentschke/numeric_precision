#!/usr/bin/env python3
"""
Evaluation of cats vs dogs model(s)
"""
import numpy as np
import sys
from tensorflow.keras import models
# collection of metrics
sys.path.append("/mnt/hd_internal/hh/projects_DS/numeric_precision/")


def get_activations(model_name, custom_objects, img_data_gen, num_step):
    """
    Uses custom metrics function as a vehicle to access outputs of last
    layer
    """
    # load model
    model = models.load_model(model_name, custom_objects)
    #model.summary()
    # preallocate values
    activation_val = np.empty((num_step, len(model.metrics_names) + 1))
    # 
    min_max_examples = [0, None, 0, None]
    raw_ix = model.metrics_names.index("raw_raw_values")
    for ix in range(num_step):
        img, label = img_data_gen.__next__()
        out = model.evaluate(img, label, verbose=0)
        activation_val[ix, :-1] = out
        # save labels in last column
        activation_val[ix, -1] = label[0][0]
        
        # save raw values and images of extremes
        if (img.shape[0] == 1) and (out[raw_ix] < min_max_examples[0]):
            min_max_examples[0] = out[raw_ix]
            min_max_examples[1] = img.reshape(img.shape[1:])
        elif (img.shape[0] == 1) and (out[raw_ix] > min_max_examples[2]):
            min_max_examples[2] = out[raw_ix]
            min_max_examples[3] = img.reshape(img.shape[1:])
        
    return activation_val, model.metrics_names, min_max_examples

def get_activations_using_predict(model_name, custom_objects, img_data_gen, num_step):
    """
    Cool for batch size = 1, but doesn't work for others
    """
    # load model
    model = models.load_model(model_name, custom_objects)
    # preallocate values
    activation_val = np.empty((num_step, 3))
    for ix in range(num_step):
        img, label = img_data_gen.__next__()
        out = model.predict(img)
        activation_val[ix, :-1] = out
        activation_val[ix, -1] = label[0]
        
    return activation_val