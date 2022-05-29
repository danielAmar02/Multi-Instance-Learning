from functions import Get_train_valid_Path
from Data_aug import DataGenerator
import numpy as np
import cv2
import skimage.io
import skimage.transform
import imageio
from funcs import *
from model import *
from train import *



input_dim = (64,64,3)

run = 1
n_folds = 4
acc = np.zeros((run, n_folds), dtype=float)
data_path = '/content/drive/MyDrive/3md3070-dlmi/trainset'

for irun in range(run):
  dataset = load_dataset(path=data_path, n_folds=n_folds, rand_state=irun)

  for ifold in tqdm(range(n_folds)):
      print ('run=', irun, '  fold=', ifold)
      acc[irun][ifold] = model_training(input_dim, dataset[ifold], irun, ifold)
print ('mi-net mean accuracy = ', np.mean(acc))
print ('std = ', np.std(acc))
