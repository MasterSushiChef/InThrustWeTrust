"""
Author: Wesley Lao
19 September 2021
"""

import numpy as np
from numpy import deg2rad, linalg as la, rad2deg
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from pyproj import Proj
from atrConstants import *

### Fuction for finding angle between vectors
def angles(vec1,vec2):
    dotp = np.dot(vec1,vec2)
    return np.arccos(dotp/la.norm(vec1)/la.norm(vec2))
# endDef

### Function for detecting yellowness
# convert to hsv and test for yellow
def yellowhsv(im,cornersize):
    # convert im to hsv space
    hsvim = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    yelmin = np.array([25,40,40])
    yelmax = np.array([80,255,255])
    mask = cv2.inRange(hsvim, yelmin, yelmax)
    cv2.imshow("mask", mask)
    # guassian blur
    # mask = cv2.GaussianBlur(mask,(cornersize,cornersize),sigmaX=int(cornersize/2),sigmaY=int(cornersize/2))
    # open and close
    # mask = closeopen(mask,int(cornersize/2))
    # mask = openclose(mask,int(cornersize/2))
    mask = openclose(mask,int(cornersize))
    # smooth edges
    print(cornersize)
    if cornersize>2:
        mask = cv2.medianBlur(mask, cornersize)
    cv2.imshow("filtered", mask)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        cv2.drawContours(im,contours,-1,(255,0,0),3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        center = np.array([int(x+w/2),int(y+h/2)])

        # if no yellow area large enough, return none
        if np.max([w,h])<20:
            return None, None, None, None, None
        
        # extract subimage for corner detection
        subim = mask[y:y+h,x:x+w]
        cv2.imshow("subimage", subim)
        
        # draw the biggest contour (c) in green
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("contour", im)
        return im, mask, contours, subim, center
    else:
        return None, None, None, None, None
# endDef

def openclose(src, kernelsize):
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    ret = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
    return ret

def closeopen(src, kernelsize):
    kernel = np.ones((kernelsize,kernelsize),int)
    ret = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)
    return ret

def frownCheck(im, cornersize):
    # find corners
    corners = cv2.goodFeaturesToTrack(im, 28, 0.5, cornersize)
    corners = np.int0(corners)
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(im,(x,y),1,[0,255,0],-1)
    if corners.shape[0]>=10:
        return im, corners, True
    else:
        return im, corners, False
# endDef

### Rotation Matrix
def rotmat(angle):
    return np.array([[np.cos(deg2rad(angle)), -1*np.sin(deg2rad(angle))],\
                     [np.sin(deg2rad(angle)), np.cos(deg2rad(angle))]])
# endDef

### Function returning geolocation of target center
def geoloc(imshape,lat,lon,altmsl,offnad,pitch,roll,head,x,y,grdmsl=600):
    ### Flat earth approximation
    # latlon to utm
    altagl = altmsl-grdmsl
    altcam = altagl + xcam*np.sin(pitch) - zcam*np.cos(pitch)
    p = Proj(proj='utm',ellps='WGS84')
    eas,nor = p(lon,lat)

    # cam offset
    camoff = np.array([0, xcam*np.cos(pitch) + zcam*np.sin(pitch)])

    # location of target
    anglex = x/(imshape[0]) - 0.5
    anglex = anglex*fov[0] + offnad + roll
    angley = y/(imshape[1]) - 0.5
    angley = angley*fov[1] + pitch

    ydist = altcam*np.tan(deg2rad(angley))
    xdist = altcam*np.tan(deg2rad(anglex))
    tarxy = np.array([xdist,ydist]) + camoff

    tarne = np.multiply(tarxy,rotmat(-1*head))

    tarlon,tarlat = p(tarne[0],tarne[1],inverse=True)
    
    return tarlon,tarlat   
# endDef

if __name__ == "__main__":
    imageDir = "C:\\Users\\wesle\\Documents\\ASE 361L\\Image Rec"
    smilefile = os.path.join(imageDir,"Smile_Wide.png")
    frownfile = os.path.join(imageDir,"Frown_Wide.png")
    files = [smilefile,frownfile]
    for file in files:
        if os.path.isfile(file):
            im = cv2.imread(file)
            # cv2.imshow("full-scale",im)
            res = 175
            im = cv2.resize(im, (int(res*im.shape[1]/im.shape[0]), res), interpolation=cv2.INTER_AREA)
            cv2.imshow("reduced",im)
            cornersize = im.shape[0]/150
            cornersize = 2**(int(np.log(cornersize)/np.log(2))-1)+1
            # cornersize = 0
            # while cornersize < 3:
            #     try:
            #         cornersize = int(im.shape[0]/150)
            #         cornersize = 2**(int(np.log(cornersize)/np.log(2))-1)+1
            #     except:
            #         im = cv2.resize(im, 2*im.shape[0:1], interpolation=cv2.INTER_AREA)
            # cv2.imshow("upscaled",im)
            print(cornersize)
            traceim, mask, contours, subim, center = yellowhsv(im, cornersize)
            print(np.max(subim.shape))
            # if subim.shape[0]<25:
            #     im = cv2.resize(im, (int(im.shape[1]*10), int(im.shape[0]*10)), interpolation=cv2.INTER_AREA)
            #     cv2.imshow("upscaled",im)
            #     cornersize = int(im.shape[0]/150)
            #     cornersize = 2**(int(np.log(cornersize)/np.log(2))-1)+1
            # test for frown
            subim, corners, frown = frownCheck(subim, cornersize)
            print('Number of Corners: %s' %corners.shape[0])
            print('Frown: %s' %frown)
            cv2.imshow("corners",subim)
            cv2.waitKey(0)
    