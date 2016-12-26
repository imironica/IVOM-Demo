from __future__ import division
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.metrics import f1_score


class MLResult(object):
 

    def __init__(self, descriptorName, labelsTest, predictedValues):
      self.labelsTest = labelsTest
      self.descriptorName = descriptorName;
      self.predictedValues = predictedValues


    def showResults(self):
        accuracy = accuracy_score(self.labelsTest, self.predictedValues);
        confusionMatrix = confusion_matrix(self.labelsTest, self.predictedValues);
 
        print 'Accuracy ', self.descriptorName, ': ', accuracy;
 
       



