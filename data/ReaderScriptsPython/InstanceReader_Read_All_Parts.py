# Display only parts
import cv2
import numpy as np
import os
import json
import Visuallization as vis
InputDir=r"../Train/"

for ff,name in enumerate(os.listdir(InputDir)): # go over all folder annotation
    print(str(ff) + ") Folder Name=" + name)

    InPath = InputDir + "//" + name + "//"
    Im = cv2.imread(InPath + "/Image.jpg") # read image
    Ignore = cv2.imread(InPath + "/Ignore.png", 0) # Read regions to ignore
    Im[:, :, 0][Ignore>0] = 0 # Mark regions to ignor
    Im[:, :, 1][Ignore>0] = 0
    Im[:, :, 2][Ignore>0] = 0

    data = json.load(open(InPath+'Data.json', 'r')) # Load data json
#-------------------------------------------------------------------------------------------------------------------------
    for contInd in data['MaterialsAndParts']: # Go over indexes of all materials and parts in vessel
             cont=data['MaterialsAndParts'][contInd] # read material instance data
             contMask = cv2.imread(InPath + cont['MaskFilePath'], 0) #Read material mask
             if not cont['IsPart']: continue # skip parts and not full phases
             I2 = Im.copy()
             I2[:, :, 0][contMask > 0] = 0 # overlay material instance on image
             I2[:, :, 1][contMask > 0] = 0
             print("##############################################################################")
             print("Material Part data")
             print(cont) # Print material data
             vis.show(np.concatenate([Im, I2],axis=1), "Image<->Material.   Press any key to contniue.     Folder:"+name+".  Material/Part index:" + "   Classes" + str(cont['All_ClassNames'])) # Display on screen