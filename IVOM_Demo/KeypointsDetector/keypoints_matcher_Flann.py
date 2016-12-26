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
detectorName =  'KAZE';

imgQuery = cv2.imread(root + '\\searched_object.jpg',0);
(kpsQuery, descsQuery) = computeFeatures(detectorName, imgQuery);
 

cam = cv2.VideoCapture(0);
 
#Open a new window
cv2.namedWindow(detectorName)

while True:
    # Read next image
    
    image = cam.read()[1];
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
         
    (kps, descs) = computeFeatures(detectorName, image);

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5);
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params);

    matches = flann.knnMatch(descsQuery, descs, k=2);

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    matchesCount = 0;
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0];
            matchesCount+=1;

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0);

    imgResult = cv2.drawMatchesKnn(imgQuery,kpsQuery,image,kps,matches,None,**draw_params)
    resultText = '';
    if(matchesCount > descsQuery.shape[1]/3):
        resultText = 'Object found';
    else:
        resultText = 'Object not found';
 

    font = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(imgResult,resultText,(10,90), font, 3,(0,255,255),2,cv2.LINE_AA);
    cv2.imshow(detectorName, imgResult);

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break