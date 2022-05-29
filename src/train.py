import numpy as np
import time
from random import shuffle
import argparse
from keras.models import Model
import glob
import scipy.misc as sci
import tensorflow as tf

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

import os
from os import path
from tqdm import tqdm

from functions import Get_train_valid_Path

def train_eval(model, train_set, irun, ifold):

    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1
    model_train_set, model_val_set = Get_train_valid_Path(train_set, train_percentage=0.8)

    train_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_val_set)

    if not path.exists('/content/Saved_model'):
      Save_model = '/content/Saved_model'
      os.mkdir(Save_model)

    model_name = '/content/Saved_model' + "_Batch_size_" + str(batch_size) + "epoch_" + "best.hd5"

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_loss', patience=10)

    callbacks = [checkpoint_fixed_name, EarlyStop]

    #weight = {0:70,1:30}
    
    if tf.test.is_gpu_available():
      with tf.device("/device:GPU:0"):
        history = model.fit_generator(train_gen, steps_per_epoch=len(model_train_set)//batch_size,
                                             epochs=40, validation_data=val_gen,
                                            validation_steps=len(model_val_set)//batch_size, callbacks=callbacks)
    else:
      history = model.fit_generator(train_gen, steps_per_epoch=len(model_train_set)//batch_size,
                                             epochs=40, validation_data=val_gen,
                                            validation_steps=len(model_val_set)//batch_size, callbacks=callbacks)

    model.save('/content/Saved_model/saved')


    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_prec = history.history['precision_m']
    val_prec = history.history['val_precision_m']

    train_BA = history.history['BA']
    val_BA = history.history['val_BA']


    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = '/content/Saved_model/' + str(irun) + '_' + str(ifold) + "_loss_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    fig = plt.figure()
    plt.plot(train_prec)
    plt.plot(val_prec)
    plt.title('precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = '/content/Saved_model/' + str(irun) + '_' + str(ifold) + "_precision_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    fig = plt.figure()
    plt.plot(train_BA)
    plt.plot(val_BA)
    plt.title('balanced accuracy')
    plt.ylabel('balanced accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = '/content/Saved_model/' + str(irun) + '_' + str(ifold) + "_BA_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    return model_name
