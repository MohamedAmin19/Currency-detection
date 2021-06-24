# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:56:40 2021

@author: Mohamed
"""
import operator
import random
import cv2 as cv
import numpy as np
import os

def findingDescriptor(images):
    descriptors = {}
    for i in images:
        descriptors[i] = []
        for img in images[i]:
            keypoint, descriptor = orb.detectAndCompute(img,None)
            descriptors[i].append(descriptor)
    return descriptors

def findingMatches(img, descriptors, threshold=50):
    keypoint2,descriptor1 = orb.detectAndCompute(img,None)
    bruteforce = cv.BFMatcher()
    matchList = {}
    for i in descriptors:
        matchList[i] = []
        for descriptor in descriptors[i]:
            matches = bruteforce.knnMatch(descriptor, descriptor1, k=2)
            real = []
            for x,y in matches:
                if x.distance <0.75 *y.distance:
                    real.append([x])
            matchList[i].append(len(real))
    classMatches = {}
    for i in descriptors:
        if len(matchList[i]) != 0:
            if max(matchList[i]) > threshold:
                return"Real"
            if max(matchList[i]) < threshold:
                return"Fake" 
    return classMatches


training = "training"
testing= "test"
images = {}
image_names = {}
classes = ["Real"]
orb = cv.ORB_create(nfeatures=1000)

for i in classes:
    image_names[i] = os.listdir(f'{training}/{i}')

for i in classes:
    images[i] = []
    for name in image_names[i]:
        img = cv.imread(f'{training}/{i}/{name}')
        canny=cv.Canny(img,100,200)
        images[i].append(canny)

descriptors = findingDescriptor(images)
testing_names = os.listdir(f'{testing}')
for name in testing_names:
    testing_img = img = cv.imread(f'{testing}/{name}')
    canny=cv.Canny(testing_img,100,200)
    matchedClasses = findingMatches(canny, descriptors)
    print(matchedClasses)
    cv.imshow(None, canny)
    cv.waitKey(0)
   
    
    
    
    

