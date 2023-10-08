# Display all vessel  instances
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
