"""
Author: Wesley Lao
12 November 2021
"""

import sys
libpath = r"C:\Users\wesle\miniconda3\envs\arduPy\Lib\site-packages" 
pypath = r"C:\Users\wesle\miniconda3\envs\arduPy\Lib" 
sys.path.append(libpath)
sys.path.append(pypath)

import numpy as np

# some imports seen online, not sure if necessary
import clr
clr.AddReference("MissionPlanner")
import MissionPlanner
clr.AddReference("MAVLink")
import MAVLink

# set True if camera rotates opposite intended direction
reverse = False

# assign camera channel
camChannel = 9

# camera installation params
centerAng = -5 # deg
rangeOfMot = 45 # deg
rollExtremes = [centerAng - rangeOfMot, centerAng + rangeOfMot]

# set true if camera control only desired while plane is armed
onlyWhileArmed = False

# printing variable for debugging
# only prints first time through
printed = False

# set true if only controlling camera
# loops the control script indefinitely,
# call controlCam in separate controller if paired with other scripts
looping = True

def controlCam(reverse, camChannel, centerAng, rangeOfMot, rollExtremes, printed):
    run = False
    if onlyWhileArmed:
        if cs.armed:
            run = True
            if not printed:
                print("ARMED, Camera Control ON")
                printed = True
        else:
            run = False
            if printed:
                print("UNARMED, Camera Control OFF")
                printed = False
    else:
        run = True
        if not printed:
            print("Camera Control ON")
            printed = True

    if run:
        if reverse:
            a = -1
        else:
            a = 1

        roll = cs.roll
        
        if roll < rollExtremes[0]:
            pwm = 1500 - a*500
        elif roll > rollExtremes[1]:
            pwm = 1500 + a*500
        else:
            pwm = 1500 + a*500*(roll-centerAng)/rangeOfMot
            
        MAV.doCommand(MAVLink.MAV_CMD.DO_SET_SERVO, camChannel, pwm, 0, 0, 0, 0, 0)
    
    return printed

print("Starting Camera Control...")

if onlyWhileArmed:
    if not cs.armed:
        print("Waiting for Plane to be armed...")

# set to neutral
MAV.doCommand(MAVLink.MAV_CMD.DO_SET_SERVO, camChannel, 1500, 0, 0, 0, 0, 0)

while looping:
    printed = controlCam(reverse, camChannel, centerAng, rangeOfMot, rollExtremes, printed)