import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow import keras
import torch
import keras.backend as K 
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

#Loss Function
def combo_loss_ring(targets, prediction):
    
    intersection = K.sum(tf.cast(targets*prediction,tf.float32),axis=(1,2))
    union = (K.sum(targets,axis=(1,2)) + K.sum(prediction,axis=(1,2)))
    dice = 1 - (2./num_ring_classes)*K.sum(intersection/union,axis=1)
    crossentropy = keras.losses.CategoricalCrossentropy()(targets,prediction)

    return dice + crossentropy

def combo_loss_wedge(targets, prediction):
    
    intersection = K.sum(tf.cast(targets*prediction,tf.float32),axis=(1,2))
    union = (K.sum(targets,axis=(1,2)) + K.sum(prediction,axis=(1,2)))
    dice = 1 - (2./num_wedge_classes)*K.sum(intersection/union,axis=1)
    crossentropy = keras.losses.CategoricalCrossentropy()(targets,prediction)

    return dice + crossentropy


def load_rings(label_name):
    # append label
    labels = np.zeros((1,patch_size,patch_size,num_ring_classes))
    label = cv2.resize(imread(label_name),(img_size,img_size),interpolation = cv2.INTER_NEAREST)
    label = to_categorical(label,num_ring_classes)
    labels[0] = label
    return labels

def load_wedges(label_name):
    # append label
    labels = np.zeros((1,patch_size,patch_size,num_wedge_classes))
    label = cv2.resize(imread(label_name),(img_size,img_size),interpolation = cv2.INTER_NEAREST)
    for i, score in enumerate(wedges):
        label[label == score] = i
    label = to_categorical(label,num_wedge_classes)
    labels[0] = label
    return labels

def load_img(img_name):
    # append image
    images = np.zeros((1,patch_size,patch_size,3))
    new_img = cv2.resize(cv2.imread(img_name),(400,400)).astype(np.float32)
    new_img -= np.mean(new_img)
    new_img /= np.var(new_img)
    images[0] = new_img

    return images

#Read the data
dataframe = pd.read_pickle(f'./labelswithscore.pkl')

d1_test = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']
d2_test = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']
test = d1_test + d2_test
test_samples = 2150

#models
ringNet = load_model('./U-Nets/ringNet.h5')
ringNet.compile(loss=combo_loss_ring, metrics='acc')
wedgeNet = load_model('./U-Nets/wedgeNet.h5')
wedgeNet.compile(loss=combo_loss_wedge, metrics='acc')
dartNet = torch.hub.load('ultralytics/yolov5','custom','/home/jacob-relle/Dokumente/AML Projekt/exp4/weights/last.pt')

'''
ring_acc = 0
ring_var = 0
ring_top = 0
ring_bottom = 1
wedge_acc = 0
wedge_var = 0
wedge_top = 0
wedge_bottom = 1
for i in range(len(dataframe)):
    if dataframe["img_folder"][i] in test:
        group = 'test'
        score = dataframe['score'][i]

        #load the data to test and validate
        image = load_img(f'/home/jacob-relle/Dokumente/AML Projekt/cropped_images/800/{group}/{dataframe["img_folder"][i]}/{dataframe["img_name"][i]}')
        ring_label = load_rings(os.path.splitext(f'/home/jacob-relle/Dokumente/AML Projekt/deepdart_data_ring/{group}/{dataframe["img_name"][i]}')[0] + '.tif')
        wedge_label = load_wedges(os.path.splitext(f'/home/jacob-relle/Dokumente/AML Projekt/deepdart_data_wedge/{group}/{dataframe["img_name"][i]}')[0] + '.tif')

        y_ring = ringNet.evaluate(image,ring_label,verbose=0)[1]
        ring_acc += y_ring
        ring_var += (y_ring - 0.959526)**2
        ring_top = max(ring_top,y_ring)
        ring_bottom = min(ring_bottom,y_ring)

        y_wedge = wedgeNet.evaluate(image,wedge_label,verbose=0)[1]
        wedge_acc += y_wedge
        wedge_var += (y_wedge - 0.986520)**2
        wedge_top = max(wedge_top,y_wedge)
        wedge_bottom = min(wedge_bottom,y_wedge)
        # Log every 5 batches.
    if i % 100 == 0:
        print(f"Progress: {(i + 1)}")
print(f' Ring acc: {ring_acc/test_samples},Variance: {ring_var/test_samples}, top:{ring_top}, bottom{ring_bottom}')
print(f'Wedge acc: {wedge_acc/test_samples}, Variance: {wedge_var/test_samples}, top:{wedge_top}, bottom:{wedge_bottom}')
'''

score_list = []
predicted_score_list = []

for i in range(len(dataframe)):
    if dataframe["img_folder"][i] in test:
        group = 'test'
        score = dataframe['score'][i]
        predicted_score = 0

        image_path = f'/home/jacob-relle/Dokumente/AML Projekt/cropped_images/800/{group}/{dataframe["img_folder"][i]}/{dataframe["img_name"][i]}'

        #load the data to test and validate
        image = load_img(image_path)
        ring = ringNet.predict(image)
        ring = np.argmax(ring,axis=3)

        wedge = wedgeNet.predict(image)
        wedge = np.argmax(wedge,axis=3)
        wedge[wedge==21] = 25
        
        total = ring*wedge
        total[total == 75] = 50
        
        dartNet_result = dartNet(image_path)
        darts = dartNet_result.xyxyn[0][:,:-1].numpy()
        for j in range(min(darts.shape[0],3)):
            if darts[j,-1] > 0.5:
                dart_pos = (int(400*(darts[j,3] + darts[j,1]) /2) , int(400*(darts[j,2] + darts[j,0]) /2))
                predicted_score += total[0,int(dart_pos[0]),int(dart_pos[1])]
                total[0,int(dart_pos[0]),int(dart_pos[1])] = 255
        score_list.append(score)
        predicted_score_list.append(predicted_score)

        print(f'P/T: {predicted_score}/{score}')

        # Log every 5 batches.
        if i % 100 == 0:
            print(f"Progress: {(i + 1)}")
