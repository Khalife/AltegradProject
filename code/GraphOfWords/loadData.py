def loadData(path,trainData):

	""" def loadData(path,kindOfData):
			path: is the text path representing the data.
			trainData: specifies the kind of data. If it is "True" then it is a train data.

			returns: a collection of documents(lists), there labels.
	"""

	with open(path ,"r") as File:
		labels = []
		documents = []
		documentNumber = 1
		for line in File:
			content = line.split('\t')
			if trainData:
				label = (documentNumber , content[0])
				labels.append(label)
				document = (documentNumber , content[1:])
				documents.append(document)
			else:
				content = line.split()
				label = (documentNumber , content[0])
				labels.append(label)
				document = (documentNumber , content[1:])
				documents.append(document)
			documentNumber += 1

	
	return {"labels" : labels , "documents" : documents}


def loadAll():

	with open('../data/r8_train_stemmed.txt' , 'r') as File:
		labels = []
		documents = []
		documentNumber = 1
		for line in File:
			content = line.split('\t')
			label = (documentNumber , content[0])
			labels.append(label)
			auxilary = content[1:][0]
			document = (documentNumber , auxilary.split())
			documents.append(document)
			documentNumber += 1

	with open('../data/r8_test_stemmed.txt','r') as File:
		for line in File:
			content = line.split()
			label = (documentNumber , content[0])
			labels.append(label)
			document = (documentNumber , content[1:])
			documents.append(document)
			documentNumber += 1

	return {"labels" : labels , "documents" : documents}