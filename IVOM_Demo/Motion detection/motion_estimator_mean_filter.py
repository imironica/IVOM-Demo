importcv2
importnumpyasnp
importscipy.misc

#Parametersofthealgorithms
threshold=40;
lstNumberOfElements=10;

#Readfromthewebcamstream
cam=cv2.VideoCapture(0);

#Openanewwindow
winName="Motion estimator Median Filter"
cv2.namedWindow(winName)

#listofframesforthe
lstFrames=cv2.cvtColor(cam.read()[1],cv2.COLOR_RGB2GRAY);
lstFrames=np.expand_dims(lstFrames,axis=2);

#Readfirstimages(forthemeanfilter):
index=1;
while(index<lstNumberOfElements):
    lstFrames=np.append(lstFrames,np.expand_dims(cv2.cvtColor(cam.read()[1],cv2.COLOR_RGB2GRAY),axis=2),axis=2);
    index=index1;


while True:
    #Computethemedianfilter
    mediumFilter=np.median(lstFrames,axis=2);
    #Readnextimage
    currentFrame=cv2.cvtColor(cam.read()[1],cv2.COLOR_RGB2GRAY);
    currentElement=np.abs(mediumFilter-currentFrame.astype(int));

    currentElement[currentElement<threshold]=-127;
    currentElement[currentElement>=threshold]=127;
    currentElement=currentElement.astype(np.int8);

    lstFrames=np.append(lstFrames,np.expand_dims(cv2.cvtColor(cam.read()[1],cv2.COLOR_RGB2GRAY),axis=2),axis=2);

    cv2.imshow(winName,currentElement);
    key=cv2.waitKey(10)
    if key==27:
        cv2.destroyWindow(winName)
        break