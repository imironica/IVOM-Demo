import cv2
import numpy as np
import scipy.misc
from scipy import ndimage

#Parameters
threshold = 20
alpha = 0.1;

#Read from the webcam stream
cam = cv2.VideoCapture(0);

#Open a new window
winName = "Motion estimator moving average"
cv2.namedWindow(winName)

#Read first images:
average = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY).astype(int);

while True:
    #get the current frame
    currentFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY);

    #compute the moving average background estimator
    average = (1-alpha) * (average.astype(float)) +  alpha * (currentFrame.astype(float));

    #compute de difference between the current frame and the background 
    currentElement = np.abs(average - currentFrame).astype(int);

    #apply the threshold
    currentElement[currentElement < threshold] = -125;
    currentElement[currentElement >= threshold] = 125;

    currentElement = currentElement.astype(np.int8);

    #show the image
    cv2.imshow(winName, currentElement)
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break
