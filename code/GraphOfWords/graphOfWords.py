from dimensialityReduction import dimensialityReduction

from labelDictionnary import labelDictionnary

from training import training

from documentTermMatrix import documentTermMatrix

import numpy as np 

from sklearn import svm

from sklearn import metrics

from loadData import loadData

path = '../data/r8_train_stemmed.txt'
trainData = True

data_train = loadData(path,trainData)

labels_train = data_train['labels']

documents_train = data_train['documents']

directed = True
weighted = False
window = 4

parameter = .003

(TWM_train , IDF_train , TW_IDF_train) = documentTermMatrix(documents_train , trainData , window , directed , weighted, parameter)

print TW_IDF_train.shape

(dictionnaryOfClasses , labelsInNumbers_train) = labelDictionnary(labels_train)

lsi = True
numberOfComponents = 100

(reducedTrainMatrix , Y_train) = dimensialityReduction(TW_IDF_train , labelsInNumbers_train , lsi , numberOfComponents)

classifier = svm.SVC()

classifier.fit(reducedTrainMatrix , Y_train)

path = '../data/r8_test_stemmed.txt'
trainData = False

data_test = loadData(path, trainData)

labels_test = data_test['labels']

documents_test = data_test['documents']

(TWM_test , IDF_test , TW_IDF_test) = documentTermMatrix(documents_test , trainData , window , directed , weighted, parameter)

(dictionnaryOfClasses , labelsInNumbers_test) = labelDictionnary(labels_test)

(reducedTestMatrix , Y_test) = dimensialityReduction(TW_IDF_test , labelsInNumbers_test , lsi , numberOfComponents)

Y_pred = classifier.predict(reducedTestMatrix )



print metrics.precision_score(Y_test, Y_pred, average='micro')
print metrics.precision_score(Y_test, Y_pred, average='macro')
print metrics.recall_score(Y_test, Y_pred, average='micro')
print metrics.recall_score(Y_test, Y_pred, average='macro')