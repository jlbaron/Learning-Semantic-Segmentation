# Evaluate semantic segmentation for the full image
#...............................Imports..................................................................
import os
import numpy as np
import cv2
import shutil
import json
import ClassesGroups
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................

PredDir = "ExampleData/Predict/"  # Prediction annotation folder
GTDir = "ExampleData/GT/" # GT annotation folder
MinMapAreaInPixels=0 # Minimum number of pixels in map to use (smaller number will be ignored)
ClassToUse = ClassesGroups.AllClasses # Classes to check
#*****************Statistics Dictionaries***********************************************************************************************************************************
FN = {} # Num false negative pixels
FP = {} # num False Positive pixels
TP = {} # Num True Positive pixels
NumImGT = {} # Total Number of pixels
NumImPrd = {} # Total number of images containing the Class

for nm in ClassToUse:
    FN[nm]=0
    FN[nm]=0
    FP[nm]=0
    TP[nm]=0
    NumImGT[nm]=0
    NumImPrd[nm]=0

#*********************************Go over all images********************************************************************************
for dr in os.listdir(PredDir):
    PrdMainDir = PredDir + "/" + dr
    GTMainDir = GTDir + "/" + dr
    PrdSemDir = PredDir + "/" + dr + "/Semantic//" # Predicted semantic folder
    GTSemDir = GTDir + "/" + dr + "/SemanticMaps//FullImage//" # GT semantic folder
    GTData = json.load(open(GTMainDir + '/Data.json', 'r'))  # Ground true data
    Image = cv2.imread(GTMainDir + "/Image.jpg")  # Load Image
    Ignore =  cv2.imread(GTMainDir + "/Ignore.png",0) # Region in the image to ignore in the evaluation
    ROI=1-Ignore
    # ============Create list of cats in ground truth or prediction annotation mask============================================================================
    ListCats = os.listdir(PrdSemDir)
    for nm in os.listdir(GTSemDir):
        if (nm  not in ListCats): ListCats.append(nm)
    #===================================================================================================================

    #==========================Generate statitics===========================================================================
    for SemFile in ListCats: # Go over all predicted and ground truth semantic map and compare them
                if  (SemFile.replace(".png", "") not in ClassToUse): continue
                nm=SemFile.replace(".png","")
            #.........................Read semantic maps from file..............................................................................
                if not os.path.exists(PrdSemDir+"/"+SemFile): # if There isnt a predicted map and there is GT map: assume empty predicted map
                    GTSemMap = (cv2.imread(GTSemDir + "/" + SemFile, 0)>0).astype(np.uint8)# Read GT
                    PrdSemMap = GTSemMap*0
                    NumImGT[nm] += 1 # Count number of GT maps where the class appear

                elif not os.path.exists(GTSemDir+"/"+SemFile):  # if There  a predicted map and there no GT map: assume empty GT map
                    PrdSemMap = (cv2.imread(PrdSemDir + "/" + SemFile, 0)>0).astype(np.uint8) # Read predicted
                    GTSemMap = PrdSemMap * 0
                else: # read both GT and Predicted map if both exist
                    PrdSemMap = (cv2.imread(PrdSemDir + "/" + SemFile, 0)>0).astype(np.uint8) # Read predicted
                    GTSemMap = (cv2.imread(GTSemDir + "/" + SemFile, 0)>0).astype(np.uint8) # Read GT map
                    NumImGT[nm] += 1 # Count number of GT maps where the class appear
            #............Calculate IOU...........................................
                if GTSemMap.sum()<MinMapAreaInPixels: GTSemMap*=bool(0) # ignore small regions
                if PrdSemMap.sum() < MinMapAreaInPixels: PrdSemMap *= bool(0)  # ignore small regions
                GTSemMap = (GTSemMap>0)*ROI # Ignore regions outside ROI
                PrdSemMap = (PrdSemMap>0)*ROI


                TPtmp = (GTSemMap * PrdSemMap).sum() # True positive
                FP[nm] += (PrdSemMap).sum() - TPtmp # False Positive sum
                FN[nm] += (GTSemMap).sum() - TPtmp #False negative sum
                TP[nm] +=  TPtmp # True postive sum
     #================================ Display statistics==============================================================
    print("\n\n***************************prediction*****************************************************************************\n")
    print("\nMinMapAreaInPixels=" + str(MinMapAreaInPixels) + "\n")

    for nm in FP:
        if NumImGT[nm]==0: continue
        IOU = TP[nm] / (TP[nm] + FN[nm] + FP[nm]+0.00001)
        Precision = TP[nm] / (TP[nm] +  FP[nm]+0.00001)
        Recall = TP[nm] / (TP[nm] + FN[nm]+0.00001)
        print("Class:\t"+nm+"\tIOU="+str(IOU)+"\tPrecision="+str(Precision)+"\tRecall="+str(Recall)+"\tNumber Of images Containing Cat=" + str(NumImGT[nm]))











