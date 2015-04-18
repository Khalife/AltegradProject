from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

import numpy as np

def  training(matrix , Y , SVM ):
	""" def  training(matrix , Y , svm ):
			matrix: is the train data
			Y: is the labels in array
			svm: is a boolean. If svm == True we perform svm otherwise we perform AdaBoostClassifier

			return: cross_validation scores
	"""

	if SVM:
		classifier = svm.SVC()
	else: 
		classifier = AdaBoostClassifier(n_estimators=300)

	scores = cross_val_score(classifier, matrix , Y , cv=5)

	return scores