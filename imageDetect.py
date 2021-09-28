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
# convert to hsv and test for yellow
def yellowhsv(im):
    # convert im to hsv space
    hsvim = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    yelmin = np.array([25,40,40])
    yelmax = np.array([80,255,255])
    mask = cv2.inRange(hsvim, yelmin, yelmax)
    cv2.imshow("mask", mask)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        cv2.drawContours(im,contours,-1,(255,0,0),3)
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        # extract subimage for corner detection
        subim = mask[y:y+h,x:x+w]
        print(np.max(subim))
        cv2.imshow("subimage", subim)
        # draw the biggest contour (c) in green
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("contour", im)
    return im, mask, contours, subim
# endDef

def frownCheck(im):
    # smooth edges
    cv2.imshow("rough", im)
    im = cv2.medianBlur(im, 5)
    cv2.imshow("filter", im)
    # find corners
    dst = cv2.cornerHarris(im,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(im,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for i in res:
        cv2.circle(im,(i[1],i[0]),3,[0,255,0],-1)
        cv2.circle(im,(i[2],i[3]),3,[0,0,255],-1)
    if res.shape[0]>24:
        return im, corners, True
    else:
        return im, corners, False
# endDef

if __name__ == "__main__":
    imageDir = "C:\\Users\\wesle\\Documents\\ASE 361L\\Image Rec"
    # imfile = os.path.join(imageDir,"Smile_Wide.png")
    imfile = os.path.join(imageDir,"Frown_Wide.png")
    if os.path.isfile(imfile):
            im = cv2.imread(imfile)
            # cv2.imshow("full-scale",im)
            smallim = cv2.resize(im, (int(im.shape[1]/6), int(im.shape[0]/6)), interpolation=cv2.INTER_AREA)
            cv2.imshow("reduced",smallim)
    traceim, mask, contours, subim = yellowhsv(smallim)
    subim, corners, frown = frownCheck(subim)
    print(frown)
    cv2.imshow("corners",subim)
    # test for frown
    cv2.waitKey(0)
    