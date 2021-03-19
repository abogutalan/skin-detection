# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
import cv2
import numpy as np

import os
import sys

# if(len(sys.argv) != 2):
#      print("ERROR: The format should be <<< python histogram_backprojection.py <image file> >>>")
#      exit()

filename = 'skin12.jpg'

def convolve(B, r):
    D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
    cv2.filter2D(B, -1, D, B)
    return B

#Loading the image and converting to HSV
image = cv2.imread('skin_Images/' + filename)
image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
model_hsv = image_hsv[225:275,625:675] # Select ROI

#Get the model histogram M
M = cv2.calcHist([model_hsv], channels=[0, 1], mask=None, 
                  histSize=[80, 256], ranges=[0, 180, 0, 256] )

#Backprojection of our original image using the model histogram M
B = cv2.calcBackProject([image_hsv], channels=[0,1], hist=M, 
                         ranges=[0,180,0,256], scale=1)

B = convolve(B, r=5)

# Get the current working directory 
#   and create result folder for processed images
result_path = os.getcwd() + "/result_Images/histogram_backprojection"
if not os.path.exists(result_path):
    os.makedirs(result_path)

#Threshold to clean the image and merging to three-channels
_, thresh = cv2.threshold(B, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite(result_path+"/roi_" + filename,cv2.cvtColor(model_hsv,cv2.COLOR_HSV2RGB))
cv2.imwrite(result_path+"/backprojection_" + filename,cv2.bitwise_and(image,image, mask = thresh))

print("Images are saved to result_Images/histogram_backprojection.")
