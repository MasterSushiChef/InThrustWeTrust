# InThrustWeTrust
ASE361L Dream Team

atr Files are used for Automatic Target Recognition (ATR).
* atrConstants: list of constants used in atr files, should be updated per actual flight parameters
* atrUtils: functions to be used in the atrScript for ATR
* atrScript: controller script to be called from Mission Planner or other mission controller

camGimbal is a script that rotates the camera to account for aircraft roll, can be called individually or in another controller with atrScript.
