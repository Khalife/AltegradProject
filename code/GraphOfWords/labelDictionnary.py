import numpy as np 

def labelDictionnary(labels):
	""" def labelDictionary(labels):
			labels: is a list of labels

			returns: a label dictionnary and a list of labels in numbers

	"""

	strippedLabels = [ y for (n,y) in labels ]
	classes = list( set(strippedLabels) )
	
	dictionnaryOfClasses = { label : classes.index(label) for label in classes }

	labelsInNumbers = [ (n , dictionnaryOfClasses[y] ) for (n,y) in labels ]

	return (dictionnaryOfClasses , labelsInNumbers)