from loadAll import loadAll 

from getTFIDF import getTFIDF

from dimensialityReduction import dimensialityReduction

from labelDictionnary import labelDictionnary

from training import training

import numpy as np 

from sklearn import svm

from sklearn import metrics

data = loadAll()

labels = data['labels']

documents = data['documents']

T=getTFIDF(documents)

(dictionnaryOfClasses , labelsInNumbers) = labelDictionnary(labels)

lsi = True
numberOfComponents = 100

(reducedMatrix , Y) = dimensialityReduction(T , labelsInNumbers , lsi , numberOfComponents)

reducedTrainMatrix = reducedMatrix[:5485]
reducedTestMatrix = reducedMatrix[5485: ]

Y_train = Y[:5485]
Y_test = Y[5485:]

classifier = svm.SVC()

classifier.fit(reducedTrainMatrix , Y_train)

Y_pred = classifier.predict(reducedTestMatrix )

print metrics.precision_score(Y_test, Y_pred, average='micro')
print metrics.precision_score(Y_test, Y_pred, average='macro')
print metrics.recall_score(Y_test, Y_pred, average='micro')
print metrics.recall_score(Y_test, Y_pred, average='macro')