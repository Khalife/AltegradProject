from dimensialityReduction import dimensialityReduction

from labelDictionnary import labelDictionnary

from training import training

import numpy as np 

from loadData import loadData


with open( 'twidf_window4_directed_weighted' ,"r") as File:
	X = np.loadtxt(File , delimiter=',')

path = '../data/r8_train_stemmed.txt'
trainData = True

data = loadData(path,trainData)

labels = data['labels']

(dictionnaryOfClasses , labelsInNumbers) = labelDictionnary(labels)

lsi = True
numberOfComponents = 100

(reducedMatrix , Y) = dimensialityReduction(X , labelsInNumbers , lsi , numberOfComponents)


svm = True

scores = training(reducedMatrix , Y , svm )

print scores['micro']
print scores['macro']