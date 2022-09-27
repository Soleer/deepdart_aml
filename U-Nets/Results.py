import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import cv2
from tensorflow import keras
from keras import backend as K
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras.models import load_model

from tifffile import imread, imwrite

#Hyperparameter
img_size=400

epochs = 1000
batch_size = 2
patch_size = img_size
num_batches = 50

rings = [ 0.,  1.,  2.,  3.]
num_ring_classes = len(rings)
ring_classes = np.arange(num_ring_classes)


def load_label(path_to_label,num_classes):

    label = cv2.resize(imread(path_to_label),(img_size,img_size),interpolation = cv2.INTER_NEAREST)
    label = to_categorical(label,num_classes)

    return label

def load_testing_data(img_names,label_names,num_classes):
    
    # Initalise mini batch
    images = np.zeros((len(img_names),patch_size,patch_size,3))
    labels = np.zeros((len(label_names),patch_size,patch_size,num_classes))

    for i, (img_name, label_name) in enumerate(zip(img_names, label_names)):

        # append label
        labels[i] = load_label(label_name)

        # append image
        new_img = cv2.resize(cv2.imread(img_name),(400,400)).astype(np.float32)
        new_img -= np.mean(new_img)
        new_img /= np.var(new_img)
        images[i] = new_img

    return images, labels

#

#Test Data
test_img_path = ''
test_label_path = ''

    # get filenames
test_img_names = sorted(glob.glob(test_img_path+'/**/*'+'.JPG', recursive=True))
test_label_names = sorted(glob.glob(test_label_path+'/**/*'+'.tif', recursive=True))

test_img_list = [test_img_names[i:i + num_batches*batch_size] for i in range(0, len(test_img_names), num_batches*batch_size)]
test_label_list = [test_label_names[i:i + num_batches*batch_size] for i in range(0, len(test_label_names), num_batches*batch_size)]

#models

ringNet = load_model('ring_best.h5')
wedgeNet = load_model('wedge_best.h5')

x_test, y_test = load_testing_data(test_img_list[0],test_label_list[0])


for i in range(0,len(test_img_list)):
        #load the data to test and validate
        x_test, y_test = load_testing_data(test_img_list[i],test_label_list[i])
        ring = ringNet.predict(x_test)
        ring = np.argmax(ring,axis=3)

        wedge = wedgeNet.predict(x_test)
        wedge = np.argmax(wedge,axis=3)



        # Log every 5 batches.
        if i % 5 == 0:
            print(f"Progress: {((i + 1) * batch_size*num_batches)}/{len(test_img_names)} Steps"  )