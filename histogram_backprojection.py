# https://docs.opencv.org/3.4/da/d7f/tutorial_back_projection.html

import cv2 as cv
import numpy as np

import os
import sys
from colorama import init, Fore


def Hist_and_Backproj(val):
    
    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range
    
    # Histogram
    hist = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    
    # Backprojection
    backproj = cv.calcBackProject([hue], [0], hist, ranges, scale=1)
  
    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(bins):
        cv.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv.FILLED)

    # Get the current working directory 
    #   and create result folder for processed images
    result_path = os.getcwd() + "/result_Images/histogram_backprojection"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    cv.imwrite(os.path.join(result_path , "backproj_" + filename), backproj)
    cv.imwrite(os.path.join(result_path , "histogram_" + filename), histImg)

    

if __name__ == "__main__":

    if(len(sys.argv) != 2):
        print("ERROR: The format should be <<< python histogram_backprojection.py script <image file> >>>")
        exit()

    filename = sys.argv[1]

    src = cv.imread('skin_Images/' + filename)

    if src is None:
        print('Could not open or find the image: ', src)
        exit(0)

    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    ch = (0, 0)
    hue = np.empty(hsv.shape, hsv.dtype)
    cv.mixChannels([hsv], [hue], ch)

    bins = 25
    Hist_and_Backproj(bins)

    print(Fore.BLUE + "The " + filename + Fore.GREEN + " image and its histogram are saved to" + Fore.CYAN + " result_Images/histogram_backprojection." + Fore.LIGHTWHITE_EX)

