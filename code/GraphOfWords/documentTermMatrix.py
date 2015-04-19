from documentWordMatrix import documentWordMatrix

import numpy as np 
from loadData import loadData
import numpy.matlib as mtlib

def documentTermMatrix(documents , train , window , directed , weighted, parameter):
	""" def documentTermMatrix(documents , window , directed , weighted):
			documents: is the set of documents loaded.
			window: is the number of words considered at a time 
			directed: is a boolean that determines if the graph is directed
			weighted: is a boolean determines if the graph is weighted

			returns: 
	"""

	(DWM , presenceMatrix , avdl, lengths , words) = documentWordMatrix(documents ,train , window , directed , weighted)
	numberOfDocuments = len(lengths)
	numberOfWords = len(words)
	termDocumentOccurence = np.sum(presenceMatrix , axis = 0)
	auxilary = numberOfDocuments * np.ones( numberOfWords )
	IDF_ = np.divide(auxilary , termDocumentOccurence)
	IDF = mtlib.repmat(IDF_ , numberOfDocuments ,1)

	pivotMatrix = np.ones((numberOfDocuments , numberOfWords))
	for length in lengths:
		documentNumber = length[0]
		documentLength = length[1]
		pivotMatrix[ documentNumber - 1 ][:] = (1 - parameter + (parameter * documentLength / avdl) ) * np.ones(numberOfWords)

	TWM = np.divide(DWM , pivotMatrix)

	TW_IDF = np.multiply(TWM , IDF)

	return (TWM , IDF , TW_IDF)