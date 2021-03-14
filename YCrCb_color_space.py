# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

# Required modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

if(len(sys.argv) != 2):
     print("ERROR: The format should be <<< python YCrCb_color_space.py <image file> >>>")
     exit()

filename = sys.argv[1]

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

# Getting the image
image = cv2.imread('skin_Images/' + filename)
imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

# Get the current working directory 
#   and create result folder for processed images
result_path = os.getcwd() + "/result_Images/YCrCb_color_space"
if not os.path.exists(result_path):
    os.makedirs(result_path)

cv2.imwrite(result_path+"/ycrcb_"+filename, np.hstack([image,skinYCrCb]))

print("Images are saved to result_Images/HSV_color_space.")
