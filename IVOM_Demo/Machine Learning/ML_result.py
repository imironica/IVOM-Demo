from __future__ import division
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class MLResult(object):
 
    def __init__(self, descriptorName, labelsTest, predictedValues):
      self.labelsTest = labelsTest
      self.descriptorName = descriptorName;
      self.predictedValues = predictedValues


    def showResults(self):
        accuracy = accuracy_score(self.labelsTest, self.predictedValues);
        confusionMatrix = confusion_matrix(self.labelsTest, self.predictedValues);
 
        print ('Accuracy {} : {}'.format(self.descriptorName, accuracy))
 
       



