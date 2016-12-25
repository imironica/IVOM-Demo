import cv2
import numpy as np
import scipy.misc
from scipy import ndimage
#Operating system libraries
import os, sys

#Parameters
threshold = 50
alpha = 0.01;
saveFrames = True;
showBackground = False;
root = os.path.dirname(os.path.realpath(__file__)) + '\\savedFrames\\';

#Read from the webcam stream
cam = cv2.VideoCapture(0);

#Open a new window
winName = "Motion estimator moving average"
cv2.namedWindow(winName)

#Read first images:
average = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY).astype(int);

index = 0;

while True:
    #get the current frame
    currentFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY);

    #compute the moving average background estimator
    average = (1-alpha) * (average.astype(float)) +  alpha * (currentFrame.astype(float));

    #compute de difference between the current frame and the background 
    currentElement = np.abs(average - currentFrame).astype(int);

    #apply the threshold
    currentElement[currentElement < threshold] = 0;
    currentElement[currentElement >= threshold] = 255;

    currentElement = currentElement.astype(np.uint8);

    #show the image
    if(showBackground == True):
        cv2.imshow( winName,  average.astype(np.uint8));
    else:
        cv2.imshow( winName, currentElement);
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break

    #Save the image
    index = index + 1;
    if(saveFrames == True):
        filename = root + str(index) + '.jpg';
        if(showBackground == True):
            scipy.misc.imsave(filename, average.astype(np.uint8));
        else:
            scipy.misc.imsave(filename, currentElement);