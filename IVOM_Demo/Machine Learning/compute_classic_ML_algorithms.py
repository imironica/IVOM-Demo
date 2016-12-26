#Operating system libraries
import os, sys

#numpy
import numpy as np

#Image libraries
from skimage.feature import hog
from scipy.misc import imread, imsave, imresize
from skimage import color

#Classification modules
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.metrics import f1_score

from ML_result import MLResult

#Don't show warnings (regarding the deprecate functions)
import warnings
warnings.filterwarnings("ignore")

####PARAMETERS#############


computeSVM = True;
computeNearestNeighbors = True;
computeSGD = True;
computeNaiveBayes = True;
computeDecisionTrees = True;
computeAdaboost = True;
computeGradientBoosting  = True;
computeRandomForest  = True;
computeExtremellyRandomForest = True;


###########################
root = os.path.dirname(os.path.realpath(__file__));
#filenameFeatures = '\\HU.csv';
#filenameFeatures = '\\ZERNIKE.csv';
filenameFeatures = '\\MOMENTS.csv';
print("\n#############################");
print("\nFilename features: " + filenameFeatures);
print("\n#############################\n");

df = pandas.read_csv(open(root + filenameFeatures));
numpyMatrix = df.as_matrix()

numberOfFeatures = numpyMatrix.shape[0];
featureSize = numpyMatrix.shape[1]-1;

descriptorsTrain = numpyMatrix[1:numberOfFeatures:2,1:featureSize];
labelsTrain = numpyMatrix[1:numberOfFeatures:2,featureSize];
 
descriptorsTest= numpyMatrix[2:numberOfFeatures:2,1:featureSize];
labelsTest = numpyMatrix[2:numberOfFeatures:2,featureSize];


#Support vector machines
#Train the model
descriptorName = 'SVM RBF'

cValues = [0.001, 0.001, 0.01, 0.1, 1, 10]
if computeSVM:
    for cValue in cValues:
        clfSVM = svm.SVC(C=cValue, cache_size=300, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False);
        clfSVM.fit(descriptorsTrain, labelsTrain); 

        #Compute the accuracy of the model
        predictedValues = [];
        for descriptor in descriptorsTest:
            valueProbability = clfSVM.predict(descriptor)[0]
            predictedValues.append(valueProbability);

        performance = MLResult(descriptorName, labelsTest,predictedValues);
        performance.showResults();
 

#=================================================================================================#
#SGD
#Train the model
if computeSGD:
    descriptorName = 'SGD'
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(descriptorsTrain, labelsTrain); 

    #Compute the accuracy of the model
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clf.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();
#=================================================================================================#


#=================================================================================================#
#Nearest neighbor
#Train the model
if computeNearestNeighbors:
    clfNB = KNeighborsClassifier(n_neighbors=3);
    clfNB.fit(descriptorsTrain, labelsTrain); 

    #Compute the accuracy of the model
    descriptorName = 'Nearest Neighbors (3) '
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clfNB.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();
#=================================================================================================#
#Naive Bayes
#Train the model
if computeNaiveBayes:
    clf = GaussianNB();
    clf.fit(descriptorsTrain, labelsTrain); 

    #Compute the accuracy of the model
    descriptorName = 'Naive Bayes '
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clf.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();

#=================================================================================================#
#Decision trees
#Train the model
if computeDecisionTrees:
    descriptorName = 'Decision Tree Classifier '
    clf = tree.DecisionTreeClassifier();
    clf.fit(descriptorsTrain, labelsTrain); 

    #Compute the accuracy of the model
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clf.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();

#=================================================================================================#
##AdaBoost
##Train the model
if computeAdaboost:
    descriptorName = 'Adaboost Classifier '
    clf = AdaBoostClassifier(n_estimators=30)
    clf.fit(descriptorsTrain, labelsTrain); 

    ##compute the accuracy of the model
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clf.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();

#=================================================================================================#
#GradientBoostingClassifier
##Train the model
if computeGradientBoosting:
    descriptorName = 'Gradient Boosting Classifier'
    clf = GradientBoostingClassifier(n_estimators=30, learning_rate=1.0, max_depth=1, random_state=0);
    clf.fit(descriptorsTrain, labelsTrain); 

    ##Compute the accuracy of the model
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clf.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();

#=================================================================================================#
#RandomForestClassifier
if computeRandomForest:
    descriptorName = 'Random Forest Classifier'
    #Train the model
    clfRF = RandomForestClassifier(n_estimators=30, criterion="gini")
    clfRF.fit(descriptorsTrain, labelsTrain); 

    #Compute the accuracy of the model
    predictedValues = [];
    for descriptor in descriptorsTest:
        valuePredicted = clfRF.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();

#ExtremellyRandomForestClassifier
if computeExtremellyRandomForest:
    descriptorName = 'Extremelly Trees Classifier'
    #Train the model
    clfRF = ExtraTreesClassifier(n_estimators=30, criterion="gini")
    clfRF.fit(descriptorsTrain, labelsTrain); 

    #Compute the accuracy of the model
    predictedValues = [];
    for descriptor in descriptorsTest:
        a = clfRF.predict(descriptor)
        valuePredicted = clfRF.predict(descriptor)[0]
        predictedValues.append(valuePredicted);

    performance = MLResult(descriptorName, labelsTest,predictedValues);
    performance.showResults();

    
 