import networkx as nx 

from loadData import loadData

import matplotlib.pyplot as plt

def extractGraph(document,window,directed,weighted):
	"""def extractGraph(documentList):
			document: is a string representing the document
			window: is the number of words considered at a time 
			weighted: is a boolean determines if the graph is weighted

			returns: a graph of words for the document.
	"""

	documentList = document.split()

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

	return G

path = '../data/r8_train_stemmed.txt'
trainData = True

data = loadData(path,trainData)

document = data['documents'][54]
window = 4
directed = True
weighted = True

G = extractGraph(document,window,directed,weighted)
print G.nodes()
sett = set(document.split())
print len(sett)
print len(G.nodes())
print G.edges()
nx.draw(G)
plt.draw()
plt.show()