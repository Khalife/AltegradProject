""" This module loads the data.
	Each line represents a document. Each line is transformed into a list of words. 
	The whole corpus is arranged into a dictionnary of labels and the documents.
"""

# Loading the data

def loadData(path,trainData):

	""" def loadData(path,kindOfData):
			path: is the text path representing the data.
			trainData: specifies the kind of data. If it is "True" then it is a train data.

		returns a collection of documents(lists) and there labels.
	"""

	with open(path ,"r") as File:
		labels = []
		documents = []
		for line in File:
			content = line.split('\t')
			if trainData:
				labels.append(content[0])
				document = content[1:]
				documents.append(document)
			else:
				documents.append(content)
		return {"labels" : labels , "documents" : documents}
