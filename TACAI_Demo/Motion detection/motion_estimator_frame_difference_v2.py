import cv2
#Operating system libraries
import os, sys

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)
 
 
cam = cv2.VideoCapture(0)

winName = "Movement Indicator"

cv2.namedWindow(winName)
# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
        

while True:
    t2 =  diffImg(t_minus, t, t_plus);
    cv2.imshow( winName, t2)
  # Read next image
    t_minus = t
    t = t_plus
    t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break

