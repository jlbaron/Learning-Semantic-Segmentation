# Apply inference to single file
# ...............................Imports..................................................................
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import Visuallization as vis
import cv2
import shutil
import json
import ClassesGroups

##################################Input paramaters#########################################################################################
# .................................Main Input parametrs...........................................................................................

#PredDir = "/media/sayhey/2T/Out/o2//"  # Prediction data folder
#GTDir = "/home/sayhey/Documents/DataZoo/LabPics2.1/Medical/Eval//" # Ground true data folder

PredDir = "ExampleData/Predict/"  # Prediction annotation folder
GTDir = "ExampleData/GT/" # GT annotation folder

#..................Input parameres...............................................................
ClassToUse = ClassesGroups.VesselClasses # Classes that should be checked
IgnoreVesselsThatAreParts = True  # ignore connectors condensers and other vessels which are kind of parts
MinPixelsInInstace = 2500 # Ignore smaller instances
MatchThresh = 0.5 # Threshold for matching segmentts
ClassAgnostic = True #True # Dont match classes when comparing instances only area (Ignore classification errors)
# ****************************************************************************************************************************************************
VesselSubDir =  "/VesselsInstances/" # SubDir were vessel instances will be located
LimitToVessel="" # Only read specific vessels
#*****************Statistics Dictionaries***********************************************************************************************************************************
if ClassAgnostic:
    NumGTInstances=0 # Total Number of GT instances
    FPall=0 # Total number of false postive for all predicted instances
FN = {} # Num false negative pixels
FP = {} # num False Positive pixels
TP = {} # Num True Positive pixels
SQ = {} # Segmentation quality
NumGT = {} # Num instances that contain this class in the GT

ClassToUse.append("Total")
for nm in ClassToUse: # statitics for each class
    FN[nm]=0
    FP[nm]=0
    TP[nm]=0
    SQ[nm] = 0
    NumGT[nm]=0

# .............Scan over all folders...................................................................
yy=0
for dr in os.listdir(GTDir):
   PrdMainDir = PredDir + "/" + dr
   GTMainDir = GTDir + "/" + dr
   GTData = json.load(open(GTMainDir + '/Data.json', 'r')) # Ground true data
   Img = cv2.imread(GTMainDir + "/Image.jpg")  # Load Image
   Ignore = cv2.imread(GTMainDir + "/Ignore.png", 0) # regions to ignore in the evaluation
   ROI = 1 - Ignore

    # .................... read all  all GT instances in vessel-----------------------------------------------------------------------------------------------
   GTMasks = []
   GTClassList = []
   GTIgnoreMasks = []
   for Vind in GTData["Vessels"]: # Loop over vessels
       GTVesData = GTData["Vessels"][Vind]
       if GTVesData['IsPart'] and IgnoreVesselsThatAreParts: continue # Ignore vessels that are  parts (condensers and connectors)
       VesMask = (cv2.imread(GTMainDir + GTVesData['MaskFilePath'], 0) > 0).astype(np.uint8)  # Read vessel instance mask
       GTMasks.append(VesMask*ROI)
       GTClass=GTVesData['All_ClassNames'] # GT instances classes
       GTClass=set(GTClass).intersection(ClassToUse) # Intersect approve classes and remove all other Classes
       GTClassList.append(GTClass)
       if LimitToVessel!="" and LimitToVessel not in GTVesData['All_ClassNames']: continue
       #if VesMask.sum() < MinPixelsInInstace: continue
 #*************************Display ******************************************************
            # I1 = Img.copy()
            # I1[:, :, 0][GTMasks[-1] > 0] = 0
            # I1[:, :, 1][GTMasks[-1] > 0] = 255
            #PrdMainDir
            # I2 = Img.copy()
            # I2[:, :, 0][GTIgnoreMasks[-1] > 0] = 0
            # I2[:, :, 1][GTIgnoreMasks[-1] > 0] = 255
            #
            # I3 = Img.copy()
            # I3[:, :, 0][VesMask > 0] = 0
            # I3[:, :, 1][VesMask > 0] = 255
            #
            # Overlay = np.concatenate([Img, I1, I2, I3,], axis=1)PrdMasks = []
#############################Load Predicted Masks#################################################
       PrdMasks = []
       PrdClassList = []

       PrdVesDir = PrdMainDir + VesselSubDir
       if os.path.exists(PrdMainDir): # If there are prediction for this vessel read them
           with open(PrdMainDir + '/InstanceClassList.json') as fp:
                PrdClassListTmp = json.load(fp) # Read all predicted instances classes
           # with open(PrdMainDir + '/InstanceClassProbability.json') as fp:
           #      PrdClassProbTmp = json.load(fp) # Read all predicted instances classes probabilities
           for InstFile in os.listdir(PrdVesDir): # Go over all predicted instances masks
                Mind = InstFile[:-4]
                PrdClass = set(PrdClassListTmp[Mind]).intersection(ClassToUse) # Limit to approves classes
                PrdClassList.append(PrdClass) # Get predicted instace classes
                PrdMsk=cv2.imread(PrdVesDir + "/" + InstFile, 0) > 0
                PrdMasks.append(PrdMsk*ROI) # Read predicted mask
       else:
           print("missing prediction for "+PrdVesDir)
                # vis.show(Overlay, "GT: " + str(GTClassList[-1]))

#============================Match GT instances to predicted instances=========================================================================

       IOUmat = np.zeros([len(PrdMasks), len(GTMasks)],dtype=np.float32)  # materix of IOU between predicted and GT instances
       PrdSumAll = {} # sum of pixels in predicted segment

       GTSumAll = {}  # sum of pixels in GT segment
#....................Fill IOU matrix...........................................................................
       for GTind in range(len(GTMasks)):  # Go over every predicted mask
            GTSumAll[GTind] = GTMasks[GTind].sum() # Mask area
            for Pind in range(len(PrdMasks)):
                if Pind not in PrdSumAll:
                        PrdSumAll[Pind] = PrdMasks[Pind].sum() # Pred mask area


                interAll = (PrdMasks[Pind] * GTMasks[GTind]).sum() #intesection of the full masks
                IOUall = interAll / ( GTSumAll[GTind] + PrdSumAll[Pind] - interAll + 0.0001)
                IOUmat[Pind,GTind]=IOUall # Add to matrix
       IOUmat[IOUmat<MatchThresh]=0 # IOU below given threshold (0.5) is set as 0
 # --------------------------Find best match between GT and predicted segmenrs -------------------------------------------------------------
       row_ind, col_ind = linear_sum_assignment(-IOUmat)  # Hungarian matching find the best matching GT segment To each predicted segment

 #......................Calcluate panoptic quality.......................................................................
 # ----------------------------------Statistics PQ without Classification (match ust areas)-----------------------------------------------
       if ClassAgnostic: # Ignore classification errors

           GTTPind = []  # indexes of GT instances that have TP match
           for i in range(len(row_ind)):  # Go over all predicted instances
               PRind = row_ind[i]  # Predicted instance
               GTind = col_ind[i]  # Best Match GT instance

               if IOUmat[PRind, GTind] == 0:  # No match for predicted segment
                   if PrdSumAll[PRind] < MinPixelsInInstace: continue  # ignore small segments
                   FPall += 1 # Add false postive

               else:  # Match for Predicted segment
                   if GTSumAll[GTind] < MinPixelsInInstace: continue  # ignore small segments
                   GTTPind.append(GTind) # List of GT instances that have match
                   for ct in GTClassList[GTind]:
                       if ct not in ClassToUse: continue
                       TP[ct] += 1
                       SQ[ct] += IOUmat[PRind, GTind]

           for GTind in range(len(GTClassList)):  # False negative GT instances that dont match predicted instances
               if GTSumAll[GTind] < MinPixelsInInstace: continue  # ignore small segments
               NumGTInstances += 1  # Count total number of GT instances in all test set

               for ct in GTClassList[GTind]:
                   if ct not in ClassToUse: continue
                   NumGT[ct] += 1  # Count number of instances with a given class in all dataset
                   if GTind not in GTTPind:  # if GT index does not match
                       FN[ct] += 1
           for ct in FP:  # Divide FP between classes according to the class abundance in the GT
               FP[ct] = NumGT[ct] / NumGTInstances * FPall

       # ----------------------------------Statistics PQ with Class (match class and not just areas)-----------------------------------------------
       else:
           GTTPind = []  # indexes of GT instances that have TP match
           for i in range(len(row_ind)):  # Go over all predicted instances
               PRind = row_ind[i]  # Predicted instance
               GTind = col_ind[i]  # Best Match GT instance

               if IOUmat[PRind, GTind] == 0:  # No match for predicted segment
                   if PrdSumAll[PRind] < MinPixelsInInstace: continue  # ignore small segments
                   for ct in PrdClassList[PRind]:
                       FP[ct] += 1
               else:  # Match for Predicted segment
                   if GTSumAll[GTind] < MinPixelsInInstace: continue  # ignore small segments
                   GTTPind.append(GTind) # Add to list of GT instances that have a matc
                   for ct in PrdClassList[PRind]:  # Compared classes of matching instances
                       if ct not in ClassToUse: continue
                       if ct in GTClassList[GTind]:  # If classes matches add to TP
                           TP[ct] += 1
                           SQ[ct] += IOUmat[PRind, GTind]
                       else:
                           FP[ct] += 1  # if classes not match add to FP
                   for ct in GTClassList[GTind]:  # False negatives Classes in GT instacne  but not in prediction
                       if ct in PrdClassList[PRind]: continue
                       if ct in ClassToUse:
                           FN[ct] += 1

           for GTind in range(len(GTClassList)):  # False negative GT instances that were not found
               if GTind in GTTPind: continue  # if instance  have  a match continue
               if GTSumAll[GTind] < MinPixelsInInstace: continue  # ignore small segments
               for ct in GTClassList[GTind]:  # Add to false negative if segment  dont have a match
                   if ct in ClassToUse:
                       FN[ct] += 1


    #===================Calculate Final and display final statitics======================================================================
       print("\n*******************************************************************************\n")
       if ClassAgnostic:
            print("\nClass Agnostic\n")
       else:
            print("\nWith Class\n")
       print("\nMinPixelsInInstace="+str(MinPixelsInInstace)+"\n")
       if LimitToVessel!="": print("Limited to vessel: "+ LimitToVessel)
       for ct in FN:
           TotalCases=TP[ct]+FP[ct]+FN[ct]
           if TotalCases>0:
               FRQ=TP[ct]/(TP[ct]+(FP[ct]+FN[ct])/2)
               FSQ=SQ[ct]/(TP[ct]+0.00001)
               print(ct+")\tPQ="+str(FRQ*FSQ)+"\tSQ="+str(FSQ)+"\tRQ="+str(FRQ)+"\tTotal Cases="+str(TotalCases))




