import cv2
import numpy as np
import scipy.misc

# Parameters of the algorithm
threshold = 40
lstNumberOfElements = 10
showBackground = True
saveFrames = False

# Read from the webcam stream
cam = cv2.VideoCapture(0);

# Open a new window (one for motion estimator and another for background estimator - if required)
winNameMotion = "Motion estimator Mean Filter"
cv2.namedWindow(winNameMotion)
if (showBackground == True):
    winNameBackground = "Background estimator Mean Filter"
    cv2.namedWindow(winNameBackground);


lstFrames = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
lstFrames = np.expand_dims(lstFrames, axis=2)

# Read first frames(for the mean filter):
index = 1;
while (index < lstNumberOfElements):
    lstFrames = np.append(lstFrames, np.expand_dims(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY), axis=2), axis=2)
    index = index + 1;

while True:
    # Compute the mean filter
    meanFilter = np.mean(lstFrames, axis=2)
    # Readnextimage
    currentFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    currentElement = np.abs(meanFilter - currentFrame.astype(int))

    currentElement[currentElement < threshold] = -127
    currentElement[currentElement >= threshold] = 127
    currentElement = currentElement.astype(np.int8)

    lstFrames = np.append(lstFrames, np.expand_dims(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY), axis=2), axis=2)

    cv2.imshow(winNameMotion, currentElement)
    if (showBackground == True):
        cv2.imshow(winNameBackground, meanFilter.astype(np.uint8))

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        if (showBackground == True):
            cv2.destroyWindow(winNameBackground)
        break

    # Save the image (if required)
    index = index + 1;
    if (saveFrames == True):
        filename = root + str(index) + '.jpg'
        if (showBackground == True):
            scipy.misc.imsave(filename, mediumFilter.astype(np.uint8))
        else:
            scipy.misc.imsave(filename, currentElement)
