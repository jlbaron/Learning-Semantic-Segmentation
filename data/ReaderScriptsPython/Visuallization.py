import cv2
import numpy as np
import json
import os
##########################################################################################
def ResizeToScreen(Im):

        h=Im.shape[0]
        w=Im.shape[1]
        r=np.min([600/h,1200/w])
        Im=cv2.resize(Im,(int(r*w),int(r*h)))
        return Im
########################################################################################
def show(Im,txt=""):
    import cv2
    cv2.destroyAllWindows()
   # print("IM text")
   # print(txt)
    cv2.imshow(txt,ResizeToScreen(Im.astype(np.uint8)))
  #  cv2.moveWindow(txt, 1, 1);
    ch=cv2.waitKey()
    cv2.destroyAllWindows()
    return ch



