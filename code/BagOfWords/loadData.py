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

	if trainData:
		with open("r8_train_stemmed.txt", "r") as trainFile:
			labels = []
			documents = []
			for line in trainFile:
				content = line.split('\t')
				labels.append(content[0])
				document = content[1:]
				documents.append(document)
			data = {"labels" : labels , "documents" : documents}

	else:
		with open("r8_test_stemmed.txt", "r") as testFile:
			labels = []
			documents = []
			for line in testFile:
				document = line.split('\t')
				documents.append(document)
			data = {"labels" : labels , "documents" : documents}

	return data
