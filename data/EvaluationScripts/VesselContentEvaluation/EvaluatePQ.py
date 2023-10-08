# Evaluate panoptic quality between prediction and GT annotation for vessel content.
# See ReadMe file for details, see input parameters for changing inputs
# Should run out of the box with the ExampleData folder
# ...............................Imports..................................................................
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import Visuallization as vis
import cv2
import shutil
import json
import ClassesGroups
import MatchPredGtVesselsIndx as MatchArrangeVessels

##################################Input paramaters#########################################################################################
# .................................Input parametrs ...........................................................................................
PredDir = "ExampleData/Predict/"  # Prediction annotation folder
GTDir = "ExampleData/GT/" # GT annotation folder
#...................................Input partameters preference..............................................................................................
MatchVessel=True# If true vessel indexes in the predicted and GT folders does not and will be matched by the script.
ClassToUse = ClassesGroups.VesselContentClasses # List of classes that will be used for the evlaution
IgnoreParts =True # Ignore instances that are parts
IgnoreSurfacePhase = True  # Ignore material instances cover surface
IgnoreScattered = True # Ignore material instances that are scattered

LimitToVessel=""#"Syringe"#IVBag"#"Tube"#, "IVBag", "DripChamber"] # Check only this vessel types if "" check all
IgnoreVesselsThatAreParts = True  # ignore connectors condensers and stuff
MinPixelsInVess = 1000 # Ignore smaller vessels (in pixels)
MinPixelsInInstace = 2500 # Ignore smaller instances (In pixels)
MatchThresh = 0.5 # Threshold for matching segmentts (IOU threshold for matching instances
ClassAgnostic = True#True # Dont match class when comparing instances only area
# ******************Additional Parameters**********************************************************************************************************************************
VesselMaskSubFolder="Vessels" # Where vessel masks will be stored (assuming that vessel matching between GT and predicted is needed)
VesselContentSubFolder="ContentInstances" # Where predicted vessel content materials and parts will be stored
#********************Match vessels index*********************************************************************
if MatchVessel:
    MatchArrangeVessels.MatchGTVesselToPredVessel(PredDir,GTDir,VesselDir="Vessels",ContentFolder=VesselContentSubFolder,ShortedContentFolder="ShortedContent") # If the vessels in the prediction input does not have the same order as the vessels in the input folder its necessary to match them first
    VesselContentSubFolder="ShortedContent"
#*****************Statistics Dictionaries***********************************************************************************************************************************
if ClassAgnostic:
    NumGTInstances=0 # Total Number of GT instances
    FPall=0 # Total number of false postive for all predicted instances
FN = {} # Num false negative pixels
FP = {} # num False Positive pixels
TP = {} # Num True Positive pixels
SQ = {} # Segmentation quality
NumGT = {} # Num instances that contain this class in the GT
NumVesPrd = {} # Total number of vessels containing the Class
ClassToUse.append("Total")
for nm in ClassToUse: # statitics for each class
    FN[nm]=0
    FP[nm]=0
    TP[nm]=0
    SQ[nm] = 0
    NumGT[nm]=0
    NumVesPrd[nm]=0
# .............Scan over all folders and perform evaluation...................................................................
for dr in os.listdir(GTDir):

    PrdMainDir = PredDir + "/" + dr + "//" + VesselContentSubFolder + "//"
    GTMainDir = GTDir + "/" + dr
    GTData = json.load(open(GTMainDir + '/Data.json', 'r')) # Ground true data
    Img = cv2.imread(GTMainDir + "/Image.jpg")  # Load Image
    # .................... read all  all GT instances in vessel-----------------------------------------------------------------------------------------------

    for Vind in GTData["Vessels"]: # Loop over vessels
        GTVesData = GTData["Vessels"][Vind]

        if GTVesData['IsPart'] and IgnoreVesselsThatAreParts: continue # For connector and condensers
        VesMask = (cv2.imread(GTMainDir + GTVesData['MaskFilePath'], 0) > 0).astype(np.uint8)  # Read vessel instance mask
        VesselCat = GTVesData['All_ClassNames'] # Read vessel class and data
        if LimitToVessel!="" and LimitToVessel not in VesselCat: continue # filter vessels by type
        if VesMask.sum() < MinPixelsInVess: continue # filter small vessel
        GTMasks = []
        GTClassList = []
        GTIgnoreMasks = []
    #.....................Get GT vessel content-----------------------------------------------------------------------------------------
        for Mind in GTVesData['VesselContentAll_Indx']: # Loop over all instances inside the vessel
            cont = GTData['MaterialsAndParts'][str(Mind)] # Data on GT instance
            if (cont['IsPart'] and IgnoreParts) or \
                    (cont['IsOnSurface'] and IgnoreSurfacePhase) or \
                    (cont['IsScattered'] and IgnoreScattered) or \
                    (cont['IsScattered'] and cont['IsOnSurface']): continue # Filter instances
            if not cont['ASegmentableInstance']: continue# Filter instances
            InsMap=(cv2.imread(GTMainDir + cont['MaskFilePath'], 0)).astype(np.uint8)  # Read gt  mask
            GTMasks.append(InsMap>0)
            GTClassList.append(cont['All_ClassNames']) # Read instance categiory
            GTClassList[-1].append("Total") # Super class for all classes
            GTIgnoreMasks.append((InsMap == 5) + (InsMap == 7)) # Region where the instance overlap and contain or beyond other instances are ignored in the evaluation
 #*************************Display ******************************************************
            # I1 = Img.copy()
            # I1[:, :, 0][GTMasks[-1] > 0] = 0
            # I1[:, :, 1][GTMasks[-1] > 0] = 255
            #
            # I2 = Img.copy()
            # I2[:, :, 0][GTIgnoreMasks[-1] > 0] = 0
            # I2[:, :, 1][GTIgnoreMasks[-1] > 0] = 255
            #
            # I3 = Img.copy()
            # I3[:, :, 0][VesMask > 0] = 0
            # I3[:, :, 1][VesMask > 0] = 255
            #
            # Overlay = np.concatenate([Img, I1, I2, I3,], axis=1)
            # vis.show(Overlay, "GT: " + str(GTClassList[-1]))
 #*******************Read Predicted instances**************************************************************************
        PrdMasks = []
        PrdClassList = []
        PrdInstDir = PrdMainDir + "/" + str(Vind)
        #PrdInstDir = PrdVesDir + "/Instance/"
        if os.path.exists(PrdInstDir ):  # If there are prediction for this vessel read them
            with open(PrdInstDir + '/InstanceClassList.json') as fp:
                PrdClassListTmp = json.load(fp) # Read all predicted instances classes
            # with open(PrdVesDir + '/InstanceClassProbability.json') as fp:
            #     PrdClassProbTmp = json.load(fp) # Read all predicted instances classes probabilities
            for InstFile in os.listdir(PrdInstDir): # Go over all predicted instances masks
                if (".png" in InstFile) or (".PNG" in InstFile):
                    Mind = InstFile[:-4]
                    PrdClassList.append(PrdClassListTmp[Mind]) # Get predicted instace classes
                    PrdClassList[-1].append("Total")
                    PrdMasks.append(cv2.imread(PrdInstDir + "/" + InstFile, 0)>0) # Read predicted instance mask
     #   else: continue

# *************************Display ******************************************************
#             I1 = Img.copy()
#             I1[:, :, 0][PrdMasks[-1] > 0] = 0
#             I1[:, :, 1][PrdMasks[-1] > 0] = 255
#
#
#             I2 = Img.copy()
#             I2[:, :, 0][VesMask > 0] = 0
#             I2[:, :, 1][VesMask > 0] = 255
#
#             Overlay = np.concatenate([Img, I1, I2, ], axis=1)
#             vis.show(Overlay, "Pred: " + str(GTClassList[-1]))
#*************************Create IOU matrix between predicted and  GT instances the goal is to find which instances match each other using hungarian matching*****************************

        IOUmat = np.zeros([len(PrdMasks), len(GTMasks)],dtype=np.float32)  # materix of IOU between predicted and GT instances
        PrdSumAll = {} # sum of pixels in predicted segment
        PrdSumCut = {} # sum of pixels in predicted segment no included region of overlap where the corrsesponding GT instance is contained or beyond other instances

        GTSumAll = {}  # sum of pixels in GT segment
        GTSumCut = {}  # sum of pixels in GT segment no included region of overlap where the instance is contained or beyond other instances

        for GTind in range(len(GTMasks)):  # Go over every predicted mask
            GTSumAll[GTind] = GTMasks[GTind].sum() # Mask area
            GTSumCut[GTind] = ((1-GTIgnoreMasks[GTind])*GTMasks[GTind]).sum() # Mask area not including area that the instance overlap other instances and contain or beyond them
            for Pind in range(len(PrdMasks)):
                if Pind not in PrdSumAll:
                        PrdSumAll[Pind] = PrdMasks[Pind].sum() # Pred mask area
                        PrdSumCut[Pind] = ((1-GTIgnoreMasks[GTind])*PrdMasks[Pind]).sum()

                interAll = (PrdMasks[Pind] * GTMasks[GTind]).sum() #intesection of the full masks
                interCut = ((1-GTIgnoreMasks[GTind])*PrdMasks[Pind] * GTMasks[GTind]).sum() # intersection with main instance area (not include area where GT contain or beyond other instances)

                IOUcut = interCut / ( GTSumCut[GTind] + PrdSumCut[Pind] - interCut + 0.0001) # IOU of important area not including area of instance beyond or contain other instances
                IOUall = interAll / ( GTSumAll[GTind] + PrdSumAll[Pind] - interAll + 0.0001) # IOU of
                IOUmat[Pind,GTind]=np.max([IOUall,IOUcut]) # Add to matrix
        IOUmat[IOUmat<MatchThresh]=0 # IOU beyond given threshold (0.5) is set as 0
        # # --------------------------Find best match -------------------------------------------------------------
        row_ind, col_ind = linear_sum_assignment(-IOUmat)  # Hungarian matching find the best matching GT segment To each predicted segment
        #---------------------------------------------------------------------------------------------

# *************************Display ******************************************************
#         for i in range(len(row_ind)):
#
#             I1 = Img.copy()
#             I1[:, :, 0][GTMasks[col_ind[i]] > 0] = 0
#             I1[:, :, 1][GTMasks[col_ind[i]] > 0] = 255
#
#             I2 = Img.copy()
#             I2[:, :, 0][PrdMasks[row_ind[i]]] = 0
#             I2[:, :, 1][PrdMasks[row_ind[i]]] = 255
#             Overlay = np.concatenate([Img, I1, I2], axis=1)
#
#             vis.show(Overlay, "Match: " + str(IOUmat[row_ind[i],col_ind[i]]))

# *****************************Collect Statistics***************************************************************


#--------------------------Class agnostic statitics------------------------------------------------------------------
        if ClassAgnostic:

            GTTPind=[] # indexes of GT instances that have TP match
            for i in range(len(row_ind)): # Go over all predicted instances
               PRind=row_ind[i] # Predicted instance
               GTind=col_ind[i] # Best Match GT instance

               if  IOUmat[PRind,GTind]==0: # No match for predicted segment
                   if PrdSumCut[PRind] < MinPixelsInInstace: continue # ignore small segments
                   FPall+=1

               else: # Match for Predicted segment
                   if GTSumCut[GTind] < MinPixelsInInstace: continue# ignore small segments
                   GTTPind.append(GTind)
                   for ct in GTClassList[GTind]:
                       if ct not in ClassToUse: continue
                       TP[ct]+=1
                       SQ[ct]+=IOUmat[PRind,GTind]

            for GTind in range(len(GTClassList)): # False negative GT instances that dont match predicted instances
                if GTSumCut[GTind] < MinPixelsInInstace: continue# ignore small segments
                NumGTInstances+=1 # Count total number of GT instances in all test set

                for ct in GTClassList[GTind]:
                   if ct not in ClassToUse: continue
                   NumGT[ct]+=1 # Count number of instances with a given class in all dataset
                   if GTind not in GTTPind: # if GT index does not match
                          FN[ct]+=1
            for ct in FP: # Divide FP between classes according to the class abundance in the GT
               FP[ct]=NumGT[ct]/NumGTInstances*FPall

#----------------------------------Statistics with Class-----------------------------------------------
        else:
            GTTPind = []  # indexes of GT instances that have TP match
            for i in range(len(row_ind)):  # Go over all predicted instances
                PRind = row_ind[i]  # Predicted instance
                GTind = col_ind[i]  # Best Match GT instance

                if IOUmat[PRind, GTind] == 0:  # No match for predicted segment
                    if PrdSumCut[PRind] < MinPixelsInInstace: continue# ignore small segments
                    for ct in PrdClassList[PRind]:
                        FP[ct] += 1
                else:  # Match for Predicted segment
                    if GTSumCut[GTind] < MinPixelsInInstace: continue# ignore small segments
                    GTTPind.append(GTind)
                    for ct in PrdClassList[PRind]: # Compared classes of matching instances
                        if ct not in ClassToUse: continue
                        if ct in GTClassList[GTind]: # If classes matches add to TP
                            TP[ct] += 1
                            SQ[ct] += IOUmat[PRind, GTind]
                        else:
                            FP[ct] += 1 # if classes not match add to FP
                    for ct in GTClassList[GTind]:  # False negatives Classes in GT instacne  but not in prediction
                        if ct in PrdClassList[PRind]: continue
                        if ct in ClassToUse:
                            FN[ct] += 1

            for GTind in range(len(GTClassList)):  # False negative GT instances that were not found
                if GTind in GTTPind: continue # if instance  have  a match continue
                if GTSumCut[GTind] < MinPixelsInInstace: continue# ignore small segments
                for ct in GTClassList[GTind]:# Add to false negative
                    if ct in ClassToUse:
                        FN[ct] += 1
#===================Calculate Final statitics and display======================================================================
        print("\n*******************************************************************************\n")
        if ClassAgnostic:
            print("\nClass Agnostic\n")
        else:
            print("\nWith Class\n")
        print("\nMinPixelsInInstace="+str(MinPixelsInInstace)+"  MinPixelsInVessel="+str(MinPixelsInVess)+"\n")
        if LimitToVessel!="": print("Limited to vessel: "+ LimitToVessel)
        for ct in FN:
           TotalCases=TP[ct]+FP[ct]+FN[ct]
           if TotalCases>0:
               FRQ=TP[ct]/(TP[ct]+(FP[ct]+FN[ct])/2)
               FSQ=SQ[ct]/(TP[ct]+0.00001)
               print(ct+")\tPQ="+str(FRQ*FSQ)+"\tSQ="+str(FSQ)+"\tRQ="+str(FRQ)+"\tTotal Cases="+str(TotalCases))




