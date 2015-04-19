from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

import numpy as np

from sklearn import metrics

def custom_precision_micro_score(y_true , y_pred):

	return metrics.precision_score(y_true, y_pred, average='micro')

def custom_precision_macro_score(y_true , y_pred):

	return metrics.precision_score(y_true, y_pred, average='macro')

def custom_recall_micro_score(y_true , y_pred):

	return metrics.recall_score(y_true, y_pred, average='micro')

def custom_recall_macro_score(y_true , y_pred):

	return metrics.recall_score(y_true, y_pred, average='macro')  



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

	precision_micro_scorer = metrics.make_scorer( custom_precision_micro_score )
	precision_macro_scorer = metrics.make_scorer( custom_precision_macro_score )
	recall_micro_scorer = metrics.make_scorer( custom_recall_micro_score )
	recall_macro_scorer = metrics.make_scorer( custom_recall_macro_score )

	precision_micro = cross_val_score(classifier, matrix , Y , cv=10 , scoring=precision_micro_scorer)
	precision_macro = cross_val_score(classifier, matrix , Y , cv=10 , scoring=precision_macro_scorer)
	recall_micro = cross_val_score(classifier, matrix , Y , cv=10 , scoring=recall_micro_scorer)
	recall_macro = cross_val_score(classifier, matrix , Y , cv=10 , scoring=recall_macro_scorer)

	return {'micro': (precision_micro , recall_micro) , 'macro' : (precision_macro , recall_macro)}