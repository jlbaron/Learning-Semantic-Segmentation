# Evaluate semantic segmentation IOU between prediction and GT annotation for vessel content.
# Assume that the vessels indexes in GT and PRedicted region do not match and find which vessels match first
# Change input parameters to run
# Should run out of the box
#...............................Imports..................................................................
import os
import numpy as np
import ClassesGroups
import Visuallization as vis
import cv2
import shutil
import json
import MatchPredGtVesselsIndx as MatchArrangeVessels
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
PredDir = "ExampleData/Predict/"  # Prediction annotation folder
GTDir = "ExampleData/GT/" # GT annotation folder

MatchVessel=True# If true vessel indexes in the predicted and GT folders does not and will be matched by the script.
MinMapAreaInPixels=1000 # Minimum number of pixels in map to use (smaller number will be ignored)
LimitToVessel=""#"Syringe"#IVBag"#"Tube"#, "IVBag", "DripChamber"] # Check only this vessel if "" check all
ClassToUse=ClassesGroups.VesselContentClasses #Class To check


# ******************Additional Parameters**********************************************************************************************************************************
VesselMaskSubFolder="Vessels" # Where vessel masks will be stored (assuming that vessel matching between GT and predicted is needed)
VesselContentSubFolder="ContentSemantic" # Where predicted vessel content materials and parts will be stored
#********************Match vessels index*********************************************************************
if MatchVessel:
    MatchArrangeVessels.MatchGTVesselToPredVessel(PredDir,GTDir,VesselDir="Vessels",ContentFolder=VesselContentSubFolder,ShortedContentFolder="ShortedContent") # If the vessels in the prediction input does not have the same order as the vessels in the input folder its necessary to match them first
    VesselContentSubFolder="ShortedContent"
#*****************Statistics Dictionaries***********************************************************************************************************************************
FN = {} # Num false negative pixels
FP = {} # num False Positive pixels
TP = {} # Num True Positive pixels
NumVesGT = {} # Total Number of pixels
NumVesPrd = {} # Total number of vessels containing the Class

for nm in ClassToUse:
    FN[nm]=0
    FN[nm]=0
    FP[nm]=0
    TP[nm]=0
    NumVesGT[nm]=0
    NumVesPrd[nm]=0

#*********************************Go over all images********************************************************************************
for dr in os.listdir(GTDir):
    print(dr)
    PrdMainDir = PredDir + "/" + dr + "//" + VesselContentSubFolder # Where vessels indexes match those of GT
    GTMainDir = GTDir + "/" + dr
    GTMaskDir = GTMainDir+'/Vessels/'
    GTData = json.load(open(GTMainDir + '/Data.json', 'r'))  # Ground true data
    for VesDir in os.listdir(GTMaskDir): # Go over all vessels in thei image
            VesDir=VesDir.replace(".png", "")
            VesselCat = GTData["Vessels"][VesDir]['All_ClassNames']
            if LimitToVessel != "" and LimitToVessel not in VesselCat: continue

            PrdSemDir = PrdMainDir + "/" + VesDir +"/"
            GTSemDir = GTMainDir + "/SemanticMaps/PerVessel//" + VesDir+"//"


          #  PrdSemDir = PrdVesDir + "/Semantic/"
            VesMask = cv2.imread(GTMaskDir + "/" + VesDir + ".png",0)
            #VesMask = cv2.imread(PrdVesDir + "/Mask.png",0) # Read vessel mask



            # Image = cv2.imread(GTMainDir + "/Image.jpg")  # Load Image
            # I1 = Image.copy()
            # I1[:, :, 0][VesMask > 0] = 0
            # I1[:, :, 1][VesMask > 0] = 0


            #============Create list of cats in ground truth or prediction annotation mask============================================================================
            ListCats=[]
            if os.path.exists(PrdSemDir):
                    ListCats=os.listdir(PrdSemDir)
            else: continue
            for nm in os.listdir(GTSemDir):
                if (not nm in ListCats) and (nm.replace(".png","") in ClassToUse): ListCats.append(nm)
            #==========================Generate statitics===========================================================================
            for SemFile in ListCats: # Go over all predicted and ground truth semantic map nd compare them
                nm=SemFile.replace(".png","")
            #.........................Read semantic maps from file..............................................................................
                if not os.path.exists(PrdSemDir+"/"+SemFile): # if There isnt a predicted map and there is GT map: assume empty pred map
                    GTSemMap = cv2.imread(GTSemDir + "/" + SemFile, 0)# Read predicted
                    PrdSemMap = GTSemMap*0
                    NumVesGT[nm] += 1 # Count number of GT maps where the class appear

                elif not os.path.exists(GTSemDir+"/"+SemFile):  # if There  a predicted map and there not GT map: assume empty GT map
                    PrdSemMap = cv2.imread(PrdSemDir + "/" + SemFile, 0)
                    GTSemMap = PrdSemMap * 0
                else: # read both GT and Predicted map if both exist
                    PrdSemMap = cv2.imread(PrdSemDir + "/" + SemFile, 0) # Read predicted
                    GTSemMap = cv2.imread(GTSemDir + "/" + SemFile, 0) # Read GT map
                    NumVesGT[nm] += 1 # Count number of GT maps where the class appear
            #............Calculate True positive false positive and false negative...........................................
                PrdSemMap = (PrdSemMap > 0) * (VesMask > 0) # Limit to inside the vessel region
                GTSemMap = (GTSemMap > 0) * (VesMask > 0) # Limit to inside the vessel region
                if GTSemMap.sum()<MinMapAreaInPixels: GTSemMap*=bool(0) # ignore small regions

                TPtmp = (GTSemMap * PrdSemMap).sum() # True positive
                FP[nm] += (PrdSemMap).sum() - TPtmp # False Positive
                FN[nm] += (GTSemMap).sum() - TPtmp #False negative
                TP[nm] +=  TPtmp
#================================ Display statistics==============================================================
print("\n\n***************************prediction*****************************************************************************\n")
print("\nMinMapAreaInPixels=" + str(MinMapAreaInPixels) + "\n")
if LimitToVessel != "": print("Limited to vessel: " + LimitToVessel)

for nm in FP:
    if NumVesGT[nm]==0: continue
    IOU = TP[nm] / (TP[nm] + FN[nm] + FP[nm]+0.00001)
    Precision = TP[nm] / (TP[nm] +  FP[nm]+0.00001)
    Recall = TP[nm] / (TP[nm] + FN[nm]+0.00001)
    print("Class:\t"+nm+"\tIOU="+str(IOU)+"\tPrecision="+str(Precision)+"\tRecall="+str(Recall)+"\tNumber Of Vessel Containing Cat=" + str(NumVesGT[nm]))












