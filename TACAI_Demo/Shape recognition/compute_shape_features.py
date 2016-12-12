#Operating system libraries
import os, sys
#numpy
import numpy as np

#Image libraries
from skimage.feature import hog
from scipy.misc import imread, imsave, imresize
from skimage import color
 
import pandas as pd
import mahotas
import pylab
import cv2
import csv
#Don't show warnings (regarding the deprecate functions)
import warnings
warnings.filterwarnings("ignore")

#Read image names from the root folder
root = os.path.dirname(os.path.realpath(__file__)) + '\\ShapeDB\\';
dirs = os.listdir(root);

featuresList = ['HU','ZERNIKE','MOMENTS'];
file_to_save = 'image_list.csv';

images = [];
descriptors = [];

#Take all the elements from the image directory and put them into a list
for file in dirs:
    images.append(file);

#Save the list of the files from the directory
df = pd.DataFrame(images, columns=["colummn"], header=False);
df.to_csv(file_to_save, index=False)

#Generate features
print ('Generate features');

for currentFeature in featuresList:
    print (currentFeature);
    print("....");

    #initiate the list of the descriptors
    descriptors = [];

    #for each image from the directory
    for file in images:

        img = imread(root + file);
        img = color.rgb2gray(img);
        print (file);

        #compute Zernike moments
        if(currentFeature == 'ZERNIKE'):
            hist = mahotas.features.zernike_moments(img, 21);

        #compute Hu moments
        if(currentFeature == 'HU'):
            hist = cv2.HuMoments(cv2.moments(img)).flatten();

        #compute shape moments
        if(currentFeature == 'MOMENTS'):
            hist = cv2.moments(img)
        
        #append the current feature
        descriptors.append(hist);

    #save feature in a csv file
    feature_filename = currentFeature + '.csv';
    df = pd.DataFrame(descriptors);
    df.to_csv(feature_filename, index=False, header=False);
