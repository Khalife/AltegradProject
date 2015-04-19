import numpy as np 

def labelDictionnary(labels):
	""" def labelDictionary(labels):
			labels: is a list of labels

			returns: a label dictionnary and a list of labels in numbers

	"""

	classes = list( set( labels) )
	
	dictionnaryOfClasses = { label : classes.index(label) for label in classes }

	labelsInNumbers = [ dictionnaryOfClasses[label]  for label in labels ]

	return (dictionnaryOfClasses , labelsInNumbers)