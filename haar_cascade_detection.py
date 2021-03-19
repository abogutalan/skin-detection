# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
import numpy as np
import cv2

import os
import sys

# if(len(sys.argv) != 2):
#      print("ERROR: The format should be <<< python haar_cascade_detection.py script <image file> >>>")
#      exit()

filename = 'skin12.jpg'

# load the required XML classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# read the image
img = cv2.imread('skin_Images/' + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find the faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    # If faces are found, it returns the positions of detected faces as Rect(x,y,w,h)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # ROI for the face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # apply eye detection on the ROI (since eyes are always on the face !!! )
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# Get the current working directory 
#   and create result folder for processed images
result_path = os.getcwd() + "/result_Images/haar_cascade"
if not os.path.exists(result_path):
    os.makedirs(result_path)

cv2.imwrite(os.path.join(result_path , filename), img)

print("The image is saved to result_Images/haar_cascade folder.")

cv2.imshow(filename,img)
cv2.waitKey(0)
cv2.destroyAllWindows()