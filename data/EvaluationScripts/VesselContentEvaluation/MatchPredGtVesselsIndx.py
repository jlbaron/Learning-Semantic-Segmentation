# Match between predicted and GT vessels
# ...............................Imports..................................................................
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import Visuallization as vis
import cv2
import shutil
import json
import ClassesGroups
MatchThresh=0.2
##################################Input paramaters#########################################################################################
# .................................Main Input parametrs...........................................................................................

#PredDir = "/media/sayhey/2T/Out/TesAll" # Ground truth data
#GTDir = "/home/sayhey/Documents/DataZoo/LabPics2.1/Medical/Eval//" # Prediction data

def MatchGTVesselToPredVessel(PredDir,GTDir,VesselDir="Vessels",ContentFolder="Content",ShortedContentFolder="ShortedContent"): # match predicted and GT vessel instances
    print("Matching vessel GT instances to Predicted instances ")
    for dr in os.listdir(GTDir):
       print(dr)
       PrdMainDir = PredDir + "/" + dr + "//"
       GTMainDir = GTDir + "/" + dr + "//"
       ContSubDirName = PrdMainDir +"//"+ ContentFolder+"//" # Original vessel content folder
       ShortSubDir = PrdMainDir + ShortedContentFolder # Where shorted vessel content folders will be stored (meaning that the vessel indexes match that of the most similar vessel in GT)
       if os.path.exists(ShortSubDir):
           shutil.rmtree(ShortSubDir)
       os.mkdir(ShortSubDir)
       GTData = json.load(open(GTMainDir + '/Data.json', 'r')) # Ground true data
     #  Img = cv2.imread(GTMainDir + "/Image.jpg")  # Load Image
       Ignore = cv2.imread(GTMainDir + "/Ignore.png", 0)
       ROI = 1 - Ignore # only region within the ROI (region of interest will be considered)

        # .................... read all  all GT instances in vessel-----------------------------------------------------------------------------------------------
       GTMasks = []
       GTIgnoreMasks = []
       GTIdx2Name={} # Name of file matches to index in GTMasks array
       for Vind in GTData["Vessels"]: # Loop over vessels
           GTVesData = GTData["Vessels"][Vind]
           VesMask = (cv2.imread(GTMainDir + GTVesData['MaskFilePath'], 0) > 0).astype(np.uint8)  # Read vessel instance mask
           GTMasks.append(VesMask*ROI)
           GTIdx2Name[len(GTMasks)-1] = GTVesData['Indx'] # Convert name of file to index in array

    #############################Load Predicted Masks#################################################
       PrdMasks = []
       PrdVesDir = PrdMainDir + VesselDir
       PrIdx2Name = {} # Match indx to file name
       for InstFile in os.listdir(PrdVesDir): # Go over all predicted instances masks
            Mind = InstFile[:-4]
            PrdMsk=cv2.imread(PrdVesDir + "/" + InstFile, 0) > 0
            PrdMasks.append(PrdMsk*ROI) # Read predicted mask
            PrIdx2Name[len(PrdMasks)-1] = InstFile.replace(".png","")
            # vis.show(Overlay, "GT: " + str(GTClassList[-1]))
    #*************************Create IOU matrix between predicted and  GT instances*****************************

       IOUmat = np.zeros([len(PrdMasks), len(GTMasks)],dtype=np.float32)  # materix of IOU between predicted and GT instances
       PrdSumAll = {} # sum of pixels in predicted segment
       GTSumAll = {}  # sum of pixels in GT segment

       for GTind in range(len(GTMasks)):  # Go over every predicted mask
            GTSumAll[GTind] = GTMasks[GTind].sum() # Mask area
            for Pind in range(len(PrdMasks)):
                if Pind not in PrdSumAll:
                        PrdSumAll[Pind] = PrdMasks[Pind].sum() # Pred mask area


                interAll = (PrdMasks[Pind] * GTMasks[GTind]).sum() #intesection of the full masks
                IOUall = interAll / ( GTSumAll[GTind] + PrdSumAll[Pind] - interAll + 0.0001)
                IOUmat[Pind,GTind]=IOUall # Add to matrix
       IOUmat[IOUmat<MatchThresh]=0 # IOU beyond given threshold (0.5) is set as 0
        # # --------------------------Find best match -------------------------------------------------------------
       row_ind, col_ind = linear_sum_assignment(-IOUmat)  # Hungarian matching find the best matching GT segment To each predicted segment
        #---------------------------------------------------------------------------------------------


    #--------------------------Rearramge predicted instances to have same indexes as GT instances------------------------------------------------------------------
       for i in range(len(row_ind)): # Go over all predicted instances
           PRind=row_ind[i] # Predicted instance
           GTind=col_ind[i] # Best Match GT instance

           shutil.copytree(ContSubDirName+"//"+PrIdx2Name[PRind]+"//",ShortSubDir+"//"+str(GTIdx2Name[GTind])+"//")
