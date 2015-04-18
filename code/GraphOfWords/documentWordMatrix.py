import networkx as nx 

from extractGraph import extractGraph

from loadData import loadData

import numpy as np 

import multiprocessing 


def documentWordMatrix(documents , window , directed , weighted):
	""" def documentWordMatrix(documents , directed , weighted):
			documents: is the set of documents loaded.
			window: is the number of words considered at a time 
			directed: is a boolean that determines if the graph is directed
			weighted: is a boolean determines if the graph is weighted

			returns: 
	"""

	#pool = multiprocessing.Pool(4)


	graphs = []
	lengths = []
	listOfWords = []
	avdl = 0

	for document in documents:
		(documentNumber , length , G) = extractGraph(document,window,directed,weighted)
		graphs.append( (documentNumber , G) )
		lengths.append( (documentNumber , length) )
		listOfWords += G.nodes()
		avdl += length

	words = list( set(listOfWords) )
	avdl /= documentNumber

	sizeOfMatrix = (documentNumber , len(words))
	DWM = np.zeros(sizeOfMatrix)
	presenceMatrix = np.zeros(sizeOfMatrix)

	for graph in graphs:
		(documentNumber , G) = graph
		for iterator in range(len(words)):
			word = words[iterator]
			if word in G.nodes():
				DWM[documentNumber - 1][iterator] = G.in_degree(word) * (not weighted) * directed + G.degree(word) * (not weighted) * (not directed) + 
				presenceMatrix[documentNumber - 1][iterator] = 1

	return (DWM , presenceMatrix , avdl, lengths , words)

<<<<<<< Updated upstream
=======

path = '../data/r8_train_stemmed.txt'
trainData = True

data = loadData(path,trainData)

documents = data['documents']

directed = True
weighted = False
window = 4
print('computing DW MATRIX')
(DWM , presenceMatrix , avdl, lengths , words) = documentWordMatrix(documents ,window , directed , weighted)




>>>>>>> Stashed changes
