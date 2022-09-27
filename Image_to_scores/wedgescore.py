from cmath import pi
import pandas as pd     
import numpy as np 
import cv2, os
from tifffile import imwrite
from PIL import Image as img


#load data and set constants
labels = pd.read_pickle(f'labels.pkl')
newcalpts = np.array([[170*np.cos(11/20*pi),170*np.sin(11/20*pi)],[170*np.cos(31/20*pi),170*np.sin(31/20*pi)],[170*np.cos(21/20*pi),170*np.sin(21/20*pi)],[170*np.cos(1/20*pi),170*np.sin(1/20*pi)]])
b = np.array([6,13,4,18,1,20,5,12,9,14,11,8,16,7,19,3,17,2,15,10])
radb = np.arange(20)*pi/10 + pi/20


def homography(oldcalpts):
    h, status = cv2.findHomography(oldcalpts, newcalpts)
    return h 

def newdarts(olddarts, h):
    newdarts = np.dot(olddarts,h.T)
    newdarts[...,0] /= newdarts[...,2]
    newdarts[...,1] /= newdarts[...,2]
    return newdarts[...,:2]

def convert_to_polarcoord(old_coords):
    polar_coords = np.zeros_like(old_coords)
    polar_coords[...,0] = np.sqrt(old_coords[...,0]**2 + old_coords[...,1]**2)
    radii = np.arccos(old_coords[...,0]/polar_coords[...,0])
    
    old_y = old_coords[...,1]
    radii[old_y < 0] = 2*pi - radii[old_y < 0]
    polar_coords[...,1] = radii
    return polar_coords

def dartscore(dart):
    scores = np.zeros(dart.shape[:2])
    #exception for 6
    scores[np.logical_or(dart[:,:,1] > 351/180*pi,dart[:,:,1] < pi/20)] = b[0]
    #label each segment
    for c in range(19):
        scores[np.logical_and(dart[:,:,1] < radb[c+1],dart[:,:,1] > radb[c])] = b[c+1]
    #label the bullseyes, triple, double, and outer rings
    scores[dart[:,:,0] < 15.9] = 25 #bullseye
    scores[dart[:,:,0] > 170]  = 0 #outer ring

    return scores

d1_val = ['d1_02_06_2020', 'd1_02_16_2020', 'd1_02_22_2020']
d1_test = ['d1_03_03_2020', 'd1_03_19_2020', 'd1_03_23_2020', 'd1_03_27_2020', 'd1_03_28_2020', 'd1_03_30_2020', 'd1_03_31_2020']

d2_val = ['d2_02_03_2021', 'd2_02_05_2021']
d2_test = ['d2_03_03_2020', 'd2_02_10_2021', 'd2_02_03_2021_2']
#new split:
val = d1_val + d2_val
test = d1_test + d2_test

for i in range(len(labels)): ##über len(labels)

    if labels["img_folder"][i] in val:
        group = 'val'
    elif labels["img_folder"][i] in test:
        group = 'test'
    else:
        group = 'train'


    oldcalpts = np.array(labels["xy"][i][0:4])
    h = homography(oldcalpts)

    ###bilder in gleichem Ordner wie Code abgelegt, eventuell Ordnerpfad einfügen, da doppelte Bildernamen auftreten könnten
    im_old = cv2.imread((f'cropped_images/800/{group}/{labels["img_folder"][i]}/{labels["img_name"][i]}'), cv2.IMREAD_GRAYSCALE) 
    im_old_arr = np.array(im_old)
    
    ###pixel in koordinaten zwischen 0,1 umgewandelt wie für calpts
    x = np.arange(800)/800
    y = np.arange(800)/800
    X, Y = np.meshgrid(x,y)
    Z = np.ones_like(X)
    image_coords = np.stack((X,Y,Z),axis=-1) 
    
    ###pixelkoordinaten in "virtuelle Scheibe" umrechnen
    board_coords = newdarts(image_coords,h)
    ###pixelkoordinaten in Polarkoordinaten der "virtuellen Scheibe" umrechnen
    polar_coords = convert_to_polarcoord(board_coords)
    
    
    ###score pro pixel berechnen
    pixelscore = dartscore(polar_coords)
    
    imwrite(f'deepdart_data/{group}/{os.path.splitext(labels["img_name"][i])[0]}.tif',pixelscore, compression='zlib')

    
    