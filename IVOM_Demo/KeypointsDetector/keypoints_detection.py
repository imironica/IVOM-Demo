import cv2 
import numpy as np
#possible values: BRISK, KAZE, SIFT, SURF, MSER, FREAK
detectorNames = [ 'BRISK', 'KAZE', 'SIFT', 'SURF', 'MSER','FAST','AKAZE'];
 
cam = cv2.VideoCapture(0);
 
#Open a new window
for detectorName in detectorNames:
    cv2.namedWindow(detectorName)

while True:
    # Read next image
    
    for detectorName in detectorNames:
        image = cam.read()[1];
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
        
        if(detectorName == "SIFT"):
            detector = cv2.xfeatures2d.SIFT_create();
            (kps, descs) = detector.detectAndCompute(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "SURF"): 
            detector = cv2.xfeatures2d.SURF_create();
            (kps, descs) = detector.detectAndCompute(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "ORB"):
            detector = cv2.ORB_create();
            (kps, descs) = detector.detectAndCompute(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "FAST"):
            detector = cv2.FastFeatureDetector_create();
            kps = detector.detect(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "MSER"):
            detector = cv2.MSER_create();
            regions = detector.detectRegions(gray, None);
            hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions];
            cv2.polylines(image, hulls, 1, (0, 255, 0));

        if(detectorName == "FREAK"):
            detector = cv2.xfeatures2d.FREAK_create();
            (kps, descs)  = detector.compute(image, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "AKAZE"):
            detector = cv2.AKAZE_create();
            (kps, descs) = detector.detectAndCompute(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "BRISK"):
            detector = cv2.BRISK_create();
            (kps, descs) = detector.detectAndCompute(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));

        if(detectorName == "KAZE"):
            detector = cv2.KAZE_create();
            (kps, descs) = detector.detectAndCompute(gray, None);
            cv2.drawKeypoints(image, kps, image, (0, 255, 0));
 
        cv2.imshow( detectorName,  image);

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break