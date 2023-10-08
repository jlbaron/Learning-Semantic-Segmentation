# Display All only   semantic maps of materials per vessel
import cv2
import numpy as np
import os
import json
import shutil
import Visuallization as vis
InputDir=r"../Train/"
############################################################################################
SemanticGroups = json.load(open("../Categories/Groups.json", 'r'))  # Semantic classes grouped to superclasses

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
    SemanticVesselDir= MainSemanticFolder+"/PerVessel//"
    #SematicAllDir =  MainSemanticFolder + "/FullImage//"
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    for vdir in  os.listdir(SemanticVesselDir): # Go over all vessels dir

         PerVesDir=SemanticVesselDir+"/"+vdir
    #********************************************************************************************************
         for name in os.listdir(PerVesDir): # Go over all semantic maps
             if name.replace(".png","") not in SemanticGroups['Material Type Class']: continue
           #  if name not in tocchange: continue
             mat=cv2.imread(PerVesDir+"/"+name) # Read  semantic map of class in vessel
             mat=mat[:,:,0]>0 # Only blue channel matter

             vesmt=cv2.imread(PerVesDir+"/Vessel.png") # Read vessel  instance map
             vesmt=vesmt[:,:,0]>0# Only blue channel matter


             I=Im.copy()

             I[:, :, 0][mat]=0 # Overlay  semantic map on image
             I[:, :, 1][mat] = 0
             I2 = Im.copy()

             I2[:, :, 0][vesmt] = 0 # Overlay vessel mask on image
             I2[:, :, 1][vesmt] = 0
             print(PerVesDir+"/"+name)
             vis.show(np.concatenate([Im,I,I2],axis=1),"Press any key to continue     Class:"+name+"     Image<->Semantic<->Vessel") # Display