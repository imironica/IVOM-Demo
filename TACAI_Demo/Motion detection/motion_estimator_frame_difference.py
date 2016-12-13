import cv2
import numpy as np
import scipy.misc
from scipy import ndimage

#Parameters
threshold = 20;
useGaussianFilter = False;
useMedianFilter = True;
useUniformFilter = False;
skipFrames = 4;

#Read from the webcam stream
cam = cv2.VideoCapture(0);

#Open a new window
winName = "Motion estimator frame difference";
cv2.namedWindow(winName);

# Read first images:
index = 0
firstFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

while True:

    #skip frames
    index = 0;
    while(index<skipFrames):
        currentFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY);
        index = index +1;
    
    #set filters - to remove the webcam noise (median / gaussian / uniform filters);
    if(useGaussianFilter == False and useMedianFilter == False):
        currentElement = np.abs(currentFrame - firstFrame);
    else:    
        if(useMedianFilter == True):
            currentElement = np.abs(ndimage.median_filter(currentFrame, 3).astype(int) - ndimage.median_filter(firstFrame, 3).astype(int)).astype(np.int8);
        if(useGaussianFilter == True):   
             currentElement = np.abs(ndimage.gaussian_filter(currentFrame, 3).astype(int) - ndimage.median_filter(firstFrame, 3).astype(int)).astype(np.int8);
        if(useUniformFilter == True):
            currentElement = np.abs(ndimage.uniform_filter(currentFrame, 3).astype(int) - ndimage.uniform_filter(firstFrame, 3).astype(int)).astype(np.int8);
    
    
    firstFrame = currentFrame
    
    #set the threshold
    currentElement[currentElement < threshold] = -127;  
    currentElement[currentElement >= threshold] = 127;

    #show the image
    cv2.imshow(winName, currentElement)
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break

 
