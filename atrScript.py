"""
Author: Wesley Lao
26 October 2021
"""

import sys
sys.path.append(r"c:\Python27\Lib\site-packages")
sys.path.append(r"c:\Python27\Lib")

import numpy as np
import atrUtils as atr

# save current frame from video feed
im = [] # TODO: save im from video
cornersize = pxontar/10
cornersize = 2**(int(np.log(cornersize)/np.log(2))-1)+1
traceim, _, _, subim, center = atr.yellowhsv(im, cornersize)

# if target found, check for frown
if traceim is not None:
    subim, corners, frown = atr.frownCheck(subim, cornersize)

    # if frown, find location; else pass
    if frown:
        tarlon, tarlat = atr.geoloc(im.shape,cs.lat,cs.lon,cs.alt,offnad,cs.pitch,cs.roll,head,x,y,grdmsl=600)
        # TODO: pull camera angle from mission planner