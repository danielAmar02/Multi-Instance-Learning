import numpy as np
import time
from random import shuffle
import argparse
from keras.models import Model
import glob
import scipy.misc as sci
import tensorflow as tf
import imageio

from keras import backend as K
#from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import matplotlib.pyplot as plt

import os

def Get_train_valid_Path(Train_set, train_percentage):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    import random
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val
  
  
  
def generate_batch(path):
        bags = []
    #for each_path in path:
        for patient_name in df['ID'].unique():
          if df[df['ID']==patient_name]['LABEL'].unique()[0]!=-1:
            name_img = []
            img = []
            img_path = glob.glob(path+'/' + patient_name +'/*.jpg')
            print(img_path)
            num_ins = len(img_path)

            label = df[df['ID']==patient_name]['LABEL'].unique()[0]

            if label == 1:
                curr_label = np.ones(num_ins,dtype=np.uint8)
            else:
                curr_label = np.zeros(num_ins, dtype=np.uint8)
            for each_img in img_path:
                img_data = np.asarray(imageio.imread(each_img), dtype=np.float32)
                #img_data -= 255
                img_data[:, :, 0] -= 123.68
                img_data[:, :, 1] -= 116.779
                img_data[:, :, 2] -= 103.939
                img_data /= 255
                #plt.imshow(img_data)
                img.append(np.expand_dims(img_data,0))
                name_img.append(each_img.split('/')[-1])
            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label, name_img))

        return bags

