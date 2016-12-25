import cv2 
import numpy as np
detector = "AKAZE"; #""
a = cv2.__version__;
cam = cv2.VideoCapture(0);
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

#Open a new window
winName = "Keypoints on webcam"
cv2.namedWindow(winName)

while True:
    # Read next image
    image = cam.read()[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.AKAZE_create()
    detector = cv2.BRISK_create()
    detector = cv2.KAZE_create()
    detector = cv2.xfeatures2d.SIFT_create();
    detector = cv2.xfeatures2d.SURF_create();
    detector = cv2.ORB_create();
    #detector = cv2.xfeatures2d.FREAK_create();
    #detector = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #detector = cv2.MSER_create()
    #detector = cv2.ORB_create()

    (kps, descs) = detector.detectAndCompute(gray, None)
 
    # draw the keypoints and show the output image
    cv2.drawKeypoints(image, kps, image, (0, 255, 0))
 
    cv2.imshow( winName,  image);

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break