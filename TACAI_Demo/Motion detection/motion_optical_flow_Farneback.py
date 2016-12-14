import cv2
import numpy as np
cam = cv2.VideoCapture(0)

#Open a new window
winName = "Motion estimator Optical Flow Farneback";
cv2.namedWindow(winName);

frame1 = cam.read()[1];
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    frame2 = cam.read()[1];
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow( winName, rgb);
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break

    prvs = next

cap.release()
cv2.destroyAllWindows()