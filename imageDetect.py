"""
Author: Wesley Lao
19 September 2021
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

### Fuction for finding angle between vectors
def angles(vec1,vec2):
    dotp = np.dot(vec1,vec2)
    return np.arccos(dotp/la.norm(vec1)/la.norm(vec2))
# endDef

### Function for detecting yellowness
def yellowness(im):
    try:
        imarr = np.array(im).astype(float)
        yellowim = np.copy(imarr)

        # define unit yellow vector in bgr space
        yellow = np.array([0,1,1])
        
        # test angle between pixel color and yellow vector
        # set to white if within pi/9 rad
        for col in range(yellowim.shape[0]):
            for row in range(yellowim.shape[1]):
                # if angles(yellow,im[col,row,:]) < np.pi/4 and im[col,row,0]/la.norm(im[col,row,:]) < 1/4:
                if angles(yellow,im[col,row,:]) < np.pi/9 and la.norm(im[col,row,:]) > la.norm(np.array([80,80,0])):
                    yellowim[col,row] = 1
                else:
                    yellowim[col,row] = 0
        yelcv = np.array(yellowim * 255, dtype = np.uint8)
        cv2.imshow("yellowness",yelcv)
    except Exception as e:
        print(e)
# endDef

### convert to hsv and test for yellow
def yellowhsv(im):
    # convert im to hsv space
    hsvim = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    yelmin = np.array([25,40,40])
    yelmax = np.array([80,255,255])
    mask = cv2.inRange(hsvim, yelmin, yelmax)
    cv2.imshow("mask", mask)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask,contours,-1,(100,100,100),3)
    # for i in range(3,7,2):
    #     kernel = np.ones((i,i),np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     kernel = np.ones((i,i),np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # cv2.imshow("noise removed", mask)
    cv2.imshow("contour", mask)

    # binarize by yellow
    pass
# endDef

if __name__ == "__main__":
    imageDir = "C:\\Users\\wesle\\Documents\\ASE 361L\\Image Rec"
    imfile = os.path.join(imageDir,"FrownCloseUp.png")
    # imfile = os.path.join(imageDir,"FrownMedium.png")
    if os.path.isfile(imfile):
            im = cv2.imread(imfile)
            cv2.imshow("full-scale",im)
            # smallim = cv2.resize(im, (int(im.shape[1]/2), int(im.shape[0]/2)), interpolation=cv2.INTER_AREA)
            # cv2.imshow("reduced",smallim)
    # yellowness(smallim)
    yellowhsv(im)
    cv2.waitKey(0)
    