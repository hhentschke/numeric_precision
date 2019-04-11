#!/usr/bin/env python3
"""
Training of binary classification model(s) (think cats vs dogs).
A few points worth mentioning:
    - the code loops over the rows of a pandas dataframe which contains training hyperparameters 
    and other parameters, including type of model to train (the dataframe was generated using 
    a notebook named 'hyperparameters_to_df' - all of this is work in progress and therefore not
    (yet) documented)
    - training, validation and test metrics will be written into a copy that dataframe and the 
    dataframe will be saved to disk after each iteration
    - the models trained here MUST have two output layers, one without any activation (outputting raw
    values), the other with sigmoid activation; variable 'loss_weights' will determine on which 
    output layer the loss will be based.
    - summarizing the points above, this is code which is designed to test the performance of
    different models/hyperparameters in an automatized way
"""
# to do: 

# - automatic detection of number of images

import click

import sys, time, os
import pandas as pd
import numpy as np
from keras import optimizers, callbacks
# models
from src.models.setup_models import convnet_01, flex_base
# learning rate schedule
from src.models.training_utilities import lr_schedule_selector
# collection of metrics
from src.models.metrics_for_keras import bce_from_raw, raw_values, sigmoids, \
    contorted_binary_crossentropy, binary_accuracy_from_raw
from src.models.data_generators import get_ImageDataGenerator


sys.path.append("/mnt/hd_internal/hh/projects_DS/numeric_precision/")

# ---------- set directories - some of these are symlinks!
train_dir = "data/cats_vs_dogs_small/train"
validation_dir = "data/cats_vs_dogs_small/validation"
test_dir = "data/cats_vs_dogs_small/test"
log_dir = "reports/logs"
report_dir = "reports"
model_dir = "models"


@click.command()
@click.argument('paramfile_in', type=click.Path(exists=True))
#@click.option('--img_format_in', default="png", type=click.Choice(['png','tif']), help='format of input image files')
def main(paramfile_in):
    """ 
    """
    # split path to and name of csv file
    paramfile_in_dir, paramfile_in_name = os.path.split(paramfile_in)
    # create name of output file:
    # time stamp as a string, to be used in file name for paramfile_out
    time_str = time.strftime("%Y_%m_%d_%H-%M", time.localtime())
    paramfile_in_name_base, paramfile_in_name_ext = os.path.splitext(paramfile_in_name)
    paramfile_out_name = paramfile_in_name_base + "_out_" + time_str + paramfile_in_name_ext
    paramfile_out = os.path.join(report_dir, paramfile_out_name)
    
    # read csv file containing training parameters
    parameters_train = pd.read_csv(paramfile_in, index_col=0)
    print(parameters_train.head())
    # a few quick sanity checks
    assert(isinstance(parameters_train.loc[0, "loss_basis"], str)), "'loss_basis' must be a str"
    assert(isinstance(parameters_train.loc[0, "do_img_augment"], np.bool_)), "'do_img_augment' must be boolean"
    assert(isinstance(parameters_train.loc[0, "do_tensorboard"], np.bool_)), "'do_tensorboard' must be boolean"
    # ...many more and better tests here
    
    #sys.exit()
    
    # number of 'episodes', that is, models to train
    num_episodes = parameters_train.shape[0]
    for episode_ix in range(num_episodes):
        print("\n\n----------------- EPISODE {} OF {} -----------------------\n\n".format(episode_ix+1, num_episodes))
        # time stamp as a string, to be used in file name for saved model and 
        # log directories for tensorboard
        time_str = time.strftime("%Y_%m_%d_%H-%M", time.localtime())
        # ---------- extract training parameters from dataframe
        batch_size = parameters_train.loc[episode_ix, "batch_size"]
        num_epochs = parameters_train.loc[episode_ix, "num_epochs"]
        do_img_augment = parameters_train.loc[episode_ix, "do_img_augment"]
        optimizer_string = parameters_train.loc[episode_ix, "optimizer"]
        lr = parameters_train.loc[episode_ix, "learn_rate"]
        if ((lr <= 0) or (lr > 1)):
            print("learning rate must be strictly positive and below 1 - setting to 0.001")
            lr = 0.001
        # now convert string to function (handle)
        optimizer = getattr(optimizers, optimizer_string)
        optimizer = optimizer(lr=lr)
        lr_schedule_mode = parameters_train.loc[episode_ix, "lr_schedule"]
        lr_schedule = lr_schedule_selector(lr_schedule_mode)
    
        # this parameter decides which of the two combinations of last layer activation 
        # and implementation of loss function shall be used for training; must be either
        # "raw" or "sigmoid"
        loss_basis = parameters_train.loc[episode_ix, "loss_basis"]
        # log with tensorboard?
        do_tensorboard = parameters_train.loc[episode_ix, "do_tensorboard"]
        # how to save model, if at all?
        save_model = parameters_train.loc[episode_ix, "save_model"]
        # ---------- other parameters defined locally
        steps_per_epoch = 2000 // batch_size
        validation_steps = 1000 // batch_size
        test_steps = 1000 // batch_size
        
        # losses, one for each of the two output layers
        loss = {"raw": bce_from_raw,
                "sigmoid": "binary_crossentropy"}
        # same for metrics
        metrics = {"raw": [binary_accuracy_from_raw, raw_values, sigmoids, contorted_binary_crossentropy],
                "sigmoid": ["accuracy"]}
        
        # set loss weights depending on model. Note that setting any of the weights to zero
        # will not result any training of the affected output layer, and thus in funny 
        # values of the metric based on that layer's output
        if loss_basis == "raw":
            loss_weights = [1.0, 0.0]
        elif loss_basis == "sigmoid":
            loss_weights = [0.0, 1.0]
        
        # string in filename of model to save
        trained_model_fn_string = loss_basis + "_epoch{}_".format(str(num_epochs))
        
        # ---------- Image data generators
        train_generator = get_ImageDataGenerator(img_dir=train_dir,
                               batch_size=batch_size, 
                               doImageTransform=do_img_augment)
        
        validation_generator = get_ImageDataGenerator(img_dir=validation_dir,
                               batch_size=batch_size, 
                               doImageTransform=False) # no augmentation for validation
        
        test_generator = get_ImageDataGenerator(img_dir=test_dir,
                               batch_size=batch_size, 
                               doImageTransform=False) # no augmentation for test
        
        # ---------- model
        model_type = parameters_train.loc[episode_ix, "model_type"]
        model_weights = parameters_train.loc[episode_ix, "model_weights"]
        # if model_weights is the str "None", convert it to None because this is what predefines
        # keras models expect if a not-pretrained model is to be used
        if model_weights == "None":
            model_weights = None
        # invoke model
        if model_type == "convnet_01":
            model, model_name = convnet_01(input_shape=(150, 150, 3))
        elif model_type in ("VGG16", "InceptionV3"):
            model, model_name = flex_base(input_shape=(150, 150, 3), base_name=model_type, weights=model_weights)
        else:
            raise Exception("bad model type")
        
        # compile
        model.compile(
                optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights,
                metrics=metrics
        )
        model.summary()
        
        # ---------- callbacks
        trained_model_fn = os.path.join(model_dir, model_name + trained_model_fn_string + time_str + ".hdf5")
        cur_callbacks = [
             callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=15, 
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=True # works in Keras and in tf.keras 2.0, but not tf.keras 1.x
             )
        ]
        cur_callbacks.append(
                callbacks.LearningRateScheduler(lr_schedule, verbose=1)
                )
        if do_tensorboard:
            cur_callbacks.append(
               callbacks.TensorBoard(
                   log_dir=os.path.join(log_dir, time_str),
                   write_images=True, # unclear yet what this does
                   update_freq='epoch', # works in Keras and in tf.keras 2.0, but not tf.keras 1.x
                   # histogram_freq=1, # unfortunately, we cannot have histograms when data generators are used
               )
            )

        if save_model == "ModelCheckpoint":
            cur_callbacks.append(
                callbacks.ModelCheckpoint(
                    trained_model_fn, 
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True
                )
            )

        # ---------- train!
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=cur_callbacks,
            verbose=True
        )

        print("\n----------------- EVALUATING -----------------------\n")
        # ---------- evaluate
        metrics_out = model.evaluate_generator(
                test_generator, 
                steps=test_steps,
                verbose=1)
        
        # ----------- write metrics to dataframe and save to file
        # overwrite model name
        _, fn = os.path.split(trained_model_fn)
        parameters_train.loc[episode_ix, "model_name"] = fn
        for m_ix, col_name in enumerate(model.metrics_names):
            # if this is the first episode, instantiate metrics columns of 
            # parameters_train with nan 
            if episode_ix == 0:
                parameters_train.loc[:, col_name] = np.nan
            # write metrics
            parameters_train.loc[episode_ix, col_name] = metrics_out[m_ix]
        # save        
        parameters_train.to_csv(paramfile_out, index=True)
        
    return None


if __name__ == '__main__':
    main()