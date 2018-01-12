import cv2
import numpy as np
import scipy.misc
# Operating system libraries
import os

# Parameters of the algorithms
threshold = 50
lstNumberOfElements = 50
saveFrames = True
showBackground = True
root = os.path.dirname(os.path.realpath(__file__)) + '\\savedFrames\\'

# Read from the webcam stream
cam = cv2.VideoCapture(0)

# Open a new window
winNameMotion = "Motion estimator Median Filter"
cv2.namedWindow(winNameMotion)

if showBackground:
    winNameBackground = "Background estimator Median Filter"
    cv2.namedWindow(winNameBackground)

# List of frames for the
lstFrames = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
lstFrames = np.expand_dims(lstFrames, axis=2)

# Read first images (for the mean filter):
index = 1
while index < lstNumberOfElements:
    lstFrames = np.append(lstFrames, np.expand_dims(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY), axis=2), axis=2)
    index = index + 1

index = 0
while True:
    # Compute the median filter
    medianFilter = np.median(lstFrames, axis=2)
    # Read next image
    currentFrame = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    currentElement = np.abs(medianFilter - currentFrame.astype(int)).astype(np.uint8)

    currentElement[currentElement < threshold] = 0
    currentElement[currentElement >= threshold] = 255

    lstFrames = np.append(lstFrames, np.expand_dims(cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY), axis=2), axis=2)

    if showBackground:
        cv2.imshow(winNameBackground, medianFilter.astype(np.uint8))

    cv2.imshow(winNameMotion, currentElement)
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winNameMotion)
        if showBackground:
            cv2.destroyWindow(winNameBackground)
        break

    # Save the image
    index = index + 1
    if saveFrames:
        filename = root + str(index) + '.jpg'
        if showBackground:
            scipy.misc.imsave(filename, medianFilter.astype(np.uint8))
        else:
            scipy.misc.imsave(filename, currentElement)
