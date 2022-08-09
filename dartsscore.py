from cmath import pi
import pandas as pd     
import numpy as np 
import cv2


def homography(oldcalpts):
    h, status = cv2.findHomography(oldcalpts, newcalpts)
    return h 

def newdarts(olddarts, h):
    for dart in olddarts: 
        i = np.where(olddarts == dart)
        dart = np.r_[dart,[1]]
        dart = np.dot(h,dart)/np.dot(h,dart)[2]
        dart = np.delete(dart, 2)
        olddarts[i] = dart
    return olddarts

def polarcoord(newdartis):
    for dart in newdartis: 
        j =np.where(newdartis == dart)
        dummy = dart.copy()
        dart[0] = np.sqrt(dart[0]**2 + dart[1]**2)
        if dart[1] < 0:
            dart[1] = 2*pi -np.arccos(dummy[0]/dart[0])
        else: 
            dart[1] = np.arccos(dummy[0]/dart[0])

        newdartis[j] = dart
    return newdartis

def dartscore(polardarts):
    a = 0 
    for dart in polardarts: 
        if dart[0] < 6.35: 
            a+=50
        if dart[0] < 15.9 and dart[0] > 6.35: 
            a+=25
        if dart[0] > 170: 
            a=a 
        if dart[1] > 351/180*pi or dart[1] < pi/20:
            if dart[0] > 15.9 and dart[0] < 99: 
                a+=6
            if dart[0] > 99 and dart[0] < 107:
                a+=18
            if dart[0] > 162 and dart[0] < 170:
                a+=12
            if dart[0] > 107 and dart[0] < 162:
                a+=6
        else:
            for c in range(19):
                if dart[1] < radb[c+1] and dart[1] > radb[c]:
                    if dart[0] > 15.9 and dart[0] < 99: 
                        a+= b[c+1]
                    if dart[0] > 99 and dart[0] < 107:
                        a+=3*b[c+1]
                    if dart[0] > 162 and dart[0] < 170:
                        a+=2*b[c+1]
                    if dart[0] > 107 and dart[0] < 162: 
                        a+=b[c+1]
    return a




labels = pd.read_pickle("labels.pkl")

newcalpts = np.array([[170*np.cos(11/20*pi),170*np.sin(11/20*pi)],[170*np.cos(31/20*pi),170*np.sin(31/20*pi)],[170*np.cos(21/20*pi),170*np.sin(21/20*pi)],[170*np.cos(1/20*pi),170*np.sin(1/20*pi)]])
b = np.array([6,13,4,18,1,20,5,12,9,14,11,8,16,7,19,3,17,2,15,10])
radb = np.zeros(20)

for j in range(0,20):
    radb[j] = pi/20 + j*pi/10




for i in range(len(labels)):
    oldcalpts = np.array(labels["xy"][i][0:4].copy())
    h = homography(oldcalpts) ###homography bestimmen aus Kalibrierungspunkten

    olddarts = np.array(labels["xy"][i][4:])
    newdartis = newdarts(olddarts, h)
    polardarts = polarcoord(newdartis)
    score = dartscore(polardarts)
    print(labels["img_name"][i],score)
    


    





