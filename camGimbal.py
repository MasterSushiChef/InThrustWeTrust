"""
Author: Wesley Lao
12 November 2021
"""
# numpy not recognized by Mission Planner env
# import numpy as np

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

# printing variable for debugging
# only prints first time through
printed = False

# set true if camera control only desired while plane is armed
onlyWhileArmed = False

print("Starting Camera Control...")

if onlyWhileArmed:
    if not cs.armed:
        print("Waiting for Plane to be armed...")

while True:
    if onlyWhileArmed:
        if cs.armed:
            run = True
            if not printed:
                print("ARMED, Camera Control ON")
                printed = True
        else:
            run = False
            MAV.doCommand(MAVLink.MAV_CMD.DO_SET_SERVO, camChannel, 1500, 0, 0, 0, 0, 0)
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


