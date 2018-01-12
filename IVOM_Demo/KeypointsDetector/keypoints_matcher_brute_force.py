import cv2
# Operating system libraries
import os

def computeFeatures(detectorName, image):
    if detectorName == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
        (kps, descs) = detector.detectAndCompute(image, None)
    if detectorName == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
        (kps, descs) = detector.detectAndCompute(image, None)

    if detectorName == "AKAZE":
        detector = cv2.AKAZE_create()
        (kps, descs) = detector.detectAndCompute(image, None)

    if detectorName == "BRISK":
        detector = cv2.BRISK_create()
        (kps, descs) = detector.detectAndCompute(image, None)

    if detectorName == "KAZE":
        detector = cv2.KAZE_create()
        (kps, descs) = detector.detectAndCompute(image, None)
    return (kps, descs)


root = os.path.dirname(os.path.realpath(__file__))
detectorName = 'SIFT'
winName = ''
imgQuery = cv2.imread(root + '\\searched_object.jpg', 0)
(kpsQuery, descsQuery) = computeFeatures(detectorName, imgQuery)

cam = cv2.VideoCapture(0)

# Open a new window
cv2.namedWindow(detectorName)

while True:
    # Read next image

    image = cam.read()[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (kps, descs) = computeFeatures(detectorName, image)

    bf = cv2.BFMatcher()

    # Match descriptors.
    if descs is not None:
        matches = bf.knnMatch(descsQuery, descs, k=2)
        # Apply ratio test
        good = [];
        matchesCount = 0;
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
                matchesCount += 1

        # cv2.drawMatchesKnn expects list of lists as matches.
        imgResult = cv2.drawMatchesKnn(imgQuery, kpsQuery, image, kps, good, outImg=None, flags=2)

        resultText = ''
        if matchesCount > descsQuery.shape[1] / 3:
            resultText = 'Object found'
        else:
            resultText = 'Object not found'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgResult, resultText, (10, 90), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(detectorName, imgResult)

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(winName)
        break
