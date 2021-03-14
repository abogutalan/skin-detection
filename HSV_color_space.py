# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

# Required modules
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

if(len(sys.argv) != 2):
     print("ERROR: The format should be <<< python HSV_color_space.py <image file> >>>")
     exit()

filename = sys.argv[1]

min_HSV = np.array([0, 58, 30], dtype = "uint8")
max_HSV = np.array([33, 255, 255], dtype = "uint8")
# Geting the image
image = cv2.imread('skin_Images/' + filename)
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)

skinHSV = cv2.bitwise_and(image, image, mask = skinRegionHSV)

# Get the current working directory 
#   and create result folder for processed images
result_path = os.getcwd() + "/result_Images/HSV_color_space"
if not os.path.exists(result_path):
    os.makedirs(result_path)

cv2.imwrite(result_path+"/hsv_"+filename, np.hstack([image, skinHSV]))

print("Images are saved to result_Images/HSV_color_space.")
