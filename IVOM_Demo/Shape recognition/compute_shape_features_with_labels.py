# Operating system libraries
import os
import numpy as np

# Image libraries
from scipy.misc import imread
from skimage import color
import pandas as pd
import mahotas
import cv2

# Don't show warnings (regarding the deprecate functions)
import warnings
warnings.filterwarnings("ignore")

# Read image names from the root folder
root = os.path.dirname(os.path.realpath(__file__)) + '\\ShapeDB\\'
dirs = os.listdir(root)
 
featuresList = ['HU','ZERNIKE','MOMENTS']
file_to_save = 'image_list.csv'

images = []
descriptors = []
labels = []

# Take all the elements from the image directory and put them into a list
indexLabel = 0
lastLabel = ''
for file in dirs:
    images.append(file)
    indexUnderscore = file.find("-")
    indexPoint = file.find(".")
    if indexUnderscore > 0:
        classValue = file[0:indexUnderscore]
        if lastLabel != classValue:
            indexLabel += 1
        labels.append(indexLabel)
        lastLabel = classValue

# Save the list of the files from the directory
df = pd.DataFrame(images, columns=["colummn"])
df.to_csv(file_to_save, index=False, header = False)

# Generate features
print ('Generate features')

for currentFeature in featuresList:
    print (currentFeature)
    print(".... ")

    # initiate the list of the descriptors
    descriptors = []

    # for each image from the directory
    indexLabel = 0
    for file in images:

        img = imread(root + file)
        img = color.rgb2gray(img)
        print (file)

        # compute Zernike moments
        if currentFeature == 'ZERNIKE':
            hist = mahotas.features.zernike_moments(img, 21)

        # compute Hu moments
        if currentFeature == 'HU':
            hist = cv2.HuMoments(cv2.moments(img)).flatten()

        # compute shape moments
        if currentFeature == 'MOMENTS':
            hist = cv2.moments(img)
        
        # append the current feature
        hist = np.append(hist, [labels[indexLabel]])
        descriptors.append(hist)
        indexLabel += 1

    # Save feature in a csv file
    feature_filename = currentFeature + '.csv'
    df = pd.DataFrame(descriptors)
    df.to_csv(feature_filename, index=False, header = False)
