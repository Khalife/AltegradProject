import networkx as nx
# from loadData import loadData
import matplotlib.pyplot as plt

def loadData(path,trainData):

	""" def loadData(path,kindOfData):
			path: is the text path representing the data.
			trainData: specifies the kind of data. If it is "True" then it is a train data.

			returns: a collection of documents(lists) and there labels.
	"""

	with open(path ,"r") as File:
		labels = []
		documents = []
		for line in File:
			content = line.split('\t')
			if trainData:
				labels.append(content[0])
				document = content[1:]
				documents.append(document[0])
			else:
				documents.append(content)

	return {"labels" : labels , "documents" : documents}

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

document = data['documents'][20]
window = 4
directed = False
weighted = True

G = extractGraph(document,window,directed,weighted)
print G.nodes()
sett = set(document.split())
print len(sett)
print len(G.nodes())
print G.edges()
deg_centrality = nx.degree_centrality(G)
eig_centrality = nx.eigenvector_centrality(G)
print(deg_centrality)
print(eig_centrality)
degs = [(v,k) for k,v in deg_centrality.iteritems()]
degs.sort()
degs.reverse()
eigs = [(v,k) for k,v in eig_centrality.iteritems()]
eigs.sort()
eigs.reverse()
print(eigs)
print(degs)
plt.figure(1)
pos=nx.spring_layout(G)
nx.draw_networkx(G,pos=pos,node_size=500,node_color='w',edge_color='y',style='dashed', font_color='k',font_weight='bold')
plt.draw()
plt.show()

