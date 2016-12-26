import cv2 
import numpy as np
#Operating system libraries
import os, sys

def computeFeatures(detectorName, image):
    if(detectorName == "SIFT"): 
        detector = cv2.xfeatures2d.SIFT_create();
        (kps, descs) = detector.detectAndCompute(image, None);
    if(detectorName == "SURF"): 
        detector = cv2.xfeatures2d.SURF_create();
        (kps, descs) = detector.detectAndCompute(image, None);
 
    if(detectorName == "AKAZE"):
        detector = cv2.AKAZE_create();
        (kps, descs) = detector.detectAndCompute(image, None);

    if(detectorName == "BRISK"):
        detector = cv2.BRISK_create();
        (kps, descs) = detector.detectAndCompute(image, None);

    if(detectorName == "KAZE"):
        detector = cv2.KAZE_create();
        (kps, descs) = detector.detectAndCompute(image, None);
    return  (kps, descs);


root = os.path.dirname(os.path.realpath(__file__))
detectorName =  'BRISK';

imgQuery = cv2.imread(root + '\\searched_object.jpg',0);
(kpsQuery, descsQuery) = computeFeatures(detectorName, imgQuery);
 

cam = cv2.VideoCapture(0);
 
#Open a new window
cv2.namedWindow(detectorName)

while True:
    # Read next image
    
    image = cam.read()[1];
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    #image = cv2.imread(root + '\\searched_object.jpg',0);
         
    (kps, descs) = computeFeatures(detectorName, image);

    bf = cv2.BFMatcher()

    # Match descriptors.
    matches = bf.knnMatch(descsQuery,descs, k=2);
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(imgQuery,kpsQuery,image,kps,good, outImg = None, flags=2)
 
    cv2.imshow( detectorName, img3);

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break