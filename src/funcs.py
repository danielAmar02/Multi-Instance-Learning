
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


from sklearn.model_selection import KFold

def load_dataset(path, n_folds, rand_state):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """

    # load datapath from path
    pos_path=[]
    neg_path=[]

    for patient_name in df['ID'].unique():

        label = df[df['ID']==patient_name]['LABEL'].unique()[0]

        if label == 1:
          pos_path += glob.glob(path+'/' + patient_name)
        if label == 0 :
          neg_path += glob.glob(path+'/' + patient_name)
        

    pos_num = len(pos_path)
    neg_num = len(neg_path)

    all_path = pos_path + neg_path

    #num_bag = len(all_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets

def generate_batch(dataset_list):
        bags = []
      
        for path in dataset_list:
          patient_name=path[-4:]
          if patient_name[0]=='/':
            patient_name=path[-3:]
          if patient_name[0]=='t':
            patient_name=path[-2:]
          print(patient_name)

          if df[df['ID']==patient_name]['LABEL'].unique()[0]!=-1:
            img = []
            img_path = glob.glob(path+'/' +'/*.jpg')
            num_ins = len(img_path)

            label = df[df['ID']==patient_name]['LABEL'].unique()[0]

            if label == 1:
                curr_label = np.ones(num_ins,dtype=np.uint8)
            else:
                curr_label = np.zeros(num_ins, dtype=np.uint8)
            for each_img in img_path:
              img_data=skimage.io.imread(each_img) 
              img_data=skimage.transform.resize(img_data, (64, 64))
              img.append(np.expand_dims(img_data,0))

            stack_img = np.concatenate(img, axis=0)
            bags.append((stack_img, curr_label))
          

        return bags
