# Display all vessel  instance and the materials and parts inside them
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
#-----------------------------------------------------------------------------------------------------------------------
    for vesInd in data["Vessels"]:# go over all vessel instances
         ves=data["Vessels"][vesInd] # Read vessel instance data
         VesMask=cv2.imread(InPath+ves['MaskFilePath'],0) # Read vesse instance mask
         I1=Im.copy()
         I1[:, :, 0][VesMask > 0] = 0 #overlay  vessel on image
         I1[:, :, 1][VesMask > 0] = 0
         print("##############################################################################")
         print("Vessel data")
         print(ves) # Print vessel data
         vis.show(np.concatenate([Im,I1],axis=1),"Image<->Vessel.     Press any key to contniue. Folder:"+name+". Vessel index:"+vesInd+" Classes:"+str(ves['All_ClassNames']))
#-------------------------------------------------------------------------------------------------------------------------
         for contInd in ves['VesselContentAll_Indx']: # Go over indexes of all materials and parts in vessel
             cont=data['MaterialsAndParts'][str(contInd)] # read material instance data
             contMask = cv2.imread(InPath + cont['MaskFilePath'], 0) #Read material mask
             I2 = Im.copy()
             I2[:, :, 0][contMask > 0] = 0 # overlay material instance on image
             I2[:, :, 1][contMask > 0] = 0
             print("##############################################################################")
             print("Material Part data")
             print(cont) # Print material data
             vis.show(np.concatenate([Im, I1, I2],axis=1), "Image<->Vessel<->Material.   Press any key to contniue.     Folder:"+name+".  Material/Part index:"+str(contInd) + "   Classes" + str(cont['All_ClassNames'])) # Display on screen