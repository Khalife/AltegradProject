import networkx as nx 

from loadData import loadData

import matplotlib.pyplot as plt

def extractGraph(document, train , window,directed,weighted):
	"""def extractGraph(documentList):
			document: is a string representing the document
			window: is the number of words considered at a time 
			directed: is a boolean that determines if the graph is directed
			weighted: is a boolean determines if the graph is weighted

			returns: a graph of words for the document.
	"""

	if train:
		documentNumber = document[0]
		document = document[1][0]
		documentList = document.split()
	else:
		documentNumber = document[0]
		documentList = document[1]

	length = len(documentList)

	if directed:
		G = nx.DiGraph()
	else:
		G = nx.Graph()

	if length > 0:
		for cursor in range(length):
			w = min(window,length - cursor)

			if w > 1:
				for iterator in range(w - 1):
					if G.has_edge(documentList[cursor] , documentList[cursor + iterator + 1]):
						G[ documentList[cursor] ][ documentList[cursor + iterator + 1] ] [ 'weight' ] += weighted*1
					else:
						G.add_edge( documentList[cursor] , documentList[cursor + iterator + 1] , weight=1)

	return (documentNumber , length , G)


def extratGraphDefault(document):
	window = 4
	directed = True
	weighted = False

	return extractGraph(document,window,directed,weighted)
