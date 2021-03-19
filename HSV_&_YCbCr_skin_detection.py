#https://github.com/CHEREF-Mehdi/SkinDetection
import cv2
import numpy as np

import os
import sys

# if(len(sys.argv) != 2):
#      print("ERROR: The format should be <<< python HSV_&_YCbCr_skin_detection.py <image file> >>>")
#      exit()

filename = 'skin12.jpg'

# read the image
img = cv2.imread('skin_Images/' + filename)

#converting from gbr to hsv color space
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#skin color range for hsv color space 
HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#converting from gbr to YCbCr color space
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#skin color range for hsv color space 
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#merge skin detection (YCbCr and hsv)
global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
global_mask=cv2.medianBlur(global_mask,3)
global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


HSV_result = cv2.bitwise_not(HSV_mask)
YCrCb_result = cv2.bitwise_not(YCrCb_mask)
global_result=cv2.bitwise_not(global_mask)

# Get the current working directory 
#   and create result folder for processed images
result_path = os.getcwd() + "/result_Images/HSV_&_YCbCr"
if not os.path.exists(result_path):
    os.makedirs(result_path)

cv2.imwrite(os.path.join(result_path , filename + "_HSV.jpg"), HSV_result)
cv2.imwrite(os.path.join(result_path , filename + "_YCbCr.jpg"), YCrCb_result)
cv2.imwrite(os.path.join(result_path , filename + "_global_result.jpg"), global_result)

print("The image is saved to result_Images/HSV_&_YCbCr folder.")


#show results
# cv2.imshow("1_HSV.jpg",HSV_result)
# cv2.imshow("2_YCbCr.jpg",YCrCb_result)
# cv2.imshow("3_global_result.jpg",global_result)
# cv2.imshow("Image.jpg",img)

# cv2.waitKey(0)
cv2.destroyAllWindows()  