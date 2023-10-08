# Display All semantic maps for only liquids
import cv2
import numpy as np
import os
import json
import shutil
import Visuallization as vis
InputDir=r"../Train/"
############################################################################################
for ff,name in enumerate(os.listdir(InputDir)): # go over all annotation folders
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$44$$$4")
    print(str(ff) + ") Folder Name=" + name)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$44$$$4")

    InPath = InputDir + "//" + name + "//"
    Im = cv2.imread(InPath + "/Image.jpg") # Read image
    Ignore = cv2.imread(InPath + "/Ignore.png", 0) # Read regions to be ignores
    Im[:, :, 0][Ignore>0] = 0 # Marked regions to ignore
    Im[:, :, 1][Ignore>0] = 0
    Im[:, :, 2][Ignore>0] = 0

    MainSemanticFolder=InPath+"/SemanticMaps/"

    SematicAllDir =  MainSemanticFolder + "/FullImage//"
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #********************************************************************************************************
    for name in os.listdir(SematicAllDir): # Go over all semantic maps
           #  if name not in tocchange: continue
             if name.replace(".png", "") != "Liquid": continue
             mat=cv2.imread(SematicAllDir+"/"+name) # Read  semantic map of class in vessel
             mat=mat[:,:,0]>0 # Only blue channel matter



             I=Im.copy()

             I[:, :, 0][mat]=0 # Overlay  semantic map on image
             I[:, :, 1][mat] = 0



             vis.show(np.concatenate([Im,I],axis=1),"Press any key to continue     Class:"+name+"     Image<->Semantic") # Display