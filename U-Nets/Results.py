import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import load_model
from tifffile import imread, imwrite

#Hyperparameter
img_size=400
patch_size = img_size

rings = [ 0.,  1.,  2.,  3.]
num_ring_classes = len(rings)
ring_classes = np.arange(num_ring_classes)

wedges = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25]
num_wedge_classes = len(wedges) 
wedge_classes = np.arange(num_wedge_classes)

def load_rings(label_name):
    # append label
    label = cv2.resize(imread(label_name),(img_size,img_size),interpolation = cv2.INTER_NEAREST)
    label = to_categorical(label,num_ring_classes)
    return label

def load_wedges(label_name):
    # append label
    label = cv2.resize(imread(label_name),(img_size,img_size),interpolation = cv2.INTER_NEAREST)
    for i, score in enumerate(wedges):
        label[label == score] = i
    label = to_categorical(label,num_wedge_classes)
    return label

def load_img(img_name):
    # append image
    new_img = cv2.resize(cv2.imread(img_name),(400,400)).astype(np.float32)
    new_img -= np.mean(new_img)
    new_img /= np.var(new_img)

    return new_img

#Read the data
dataframe = pd.read_pickle(f'labels_withscore.pkl')

d1_test = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']
d2_test = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']
test = d1_test + d2_test

#models
ringNet = load_model('ring_best.h5')
wedgeNet = load_model('wedge_best.h5')

for i in range(len(dataframe)):
    if dataframe["img_folder"][i] in test:
        group = test
        score = score

        #load the data to test and validate
        image = load_img(f'cropped_images/800/{group}/{dataframe["img_folder"][i]}/{dataframe["img_name"][i]}')
        ring_label = load_rings(f'deepdart_data_ring{group}/{dataframe["img_folder"][i]}/{dataframe["img_name"][i]}')
        wedge_label = load_wedges(f'deepdart_data_wedge{group}/{dataframe["img_folder"][i]}/{dataframe["img_name"][i]}')

        ring = ringNet.predict(image)
        ring = np.argmax(ring,axis=3)

        wedge = wedgeNet.predict(image)
        wedge = np.argmax(wedge,axis=3)
        wedge[wedge==21] = 25

        total = ring*wedge
        total[total == 75] = 50

        

        # Log every 5 batches.
        if i % 100 == 0:
            print(f"Progress: {(i + 1)}")