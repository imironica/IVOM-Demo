import cv2
import numpy as np
import scipy.misc

threshold = 10;
lstNumberOfElements = 30;

cam = cv2.VideoCapture(0)
winName = "Motion estimator Mean Filter"
cv2.namedWindow(winName)

lstLastFrames = [];

# Read first images:
index = 0;
while(index < lstNumberOfElements):
    lstLastFrames.append(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY));
    index = index + 1;


while True:
    mediumFilter = np.matrix(lstLastFrames[0]).astype(int);
    for i in range(1,10):
        mediumFilter = np.matrix(mediumFilter) + np.matrix(lstLastFrames[i]).astype(int);
    mediumFilter = np.divide(mediumFilter, lstNumberOfElements).astype(int);

    # Read next image
    del lstLastFrames[0];
    currentFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY);
    lstLastFrames.append(currentFrame);

    currentElement = np.abs(mediumFilter - currentFrame.astype(int));
  
    currentElement[currentElement < threshold] = -120;  
    currentElement[currentElement >= threshold] = 120;
    currentElement = currentElement.astype(np.int8);
    cv2.imshow( winName, currentElement );
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break

 
