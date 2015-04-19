from dimensialityReduction import dimensialityReduction

from labelDictionnary import labelDictionnary

from training import training

from documentTermMatrix import documentTermMatrix

import numpy as np 

from sklearn import svm

from sklearn import metrics

from loadData import loadAll


data = loadAll()

labels = data['labels']

documents = data['documents']


directed = True
weighted = False
window = 4

parameter = .003

trainData = False


(dictionnaryOfClasses , labelsInNumbers) = labelDictionnary(labels)

(TWM , IDF , TW_IDF) = documentTermMatrix(documents , trainData , window , directed , weighted, parameter)

lsi = True
numberOfComponents = 100

(reducedMatrix , Y) = dimensialityReduction(TW_IDF , labelsInNumbers , lsi , numberOfComponents)

reducedTrainMatrix = reducedMatrix[:5485]
reducedTestMatrix = reducedMatrix[5485: ]

Y_train = Y[:5485]
Y_test = Y[5485:]

classifier = svm.SVC(kernel='rbf')

classifier.fit(reducedTrainMatrix , Y_train)

Y_pred = classifier.predict(reducedTestMatrix )

print metrics.precision_score(Y_test, Y_pred, average='micro')
print metrics.precision_score(Y_test, Y_pred, average='macro')
print metrics.recall_score(Y_test, Y_pred, average='micro')
print metrics.recall_score(Y_test, Y_pred, average='macro')
