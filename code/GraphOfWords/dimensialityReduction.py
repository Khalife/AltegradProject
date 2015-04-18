from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBes

import numpy as np

def dimensialityReduction(DTM , labels , lsi , numberOfComponents):
	"""def dimensialityReduction(DTM , lsi , numberOfComponents):
			DTM: is the document term matrix
			labels: is the list of labels
			lsi: is a boolean. If it is True we preform latent semantics indexing. Otherwise, we perform the Chi square.
			numberOfComponents: is a parameter. Indicates the new dimension.

			return: a reduced dimension matrix
	"""

	Y_ = np.zeros((1,len(labels))
	Y_[0] = np.array( [ y for (n,y) in labels ] )
	Y = Y_.transpose()


	if lsi:
		svd = TruncatedSVD(n_components = numberOfComponents)
		reducedMatrix = svd.fit_transform(DTM)
	else:
		reducedMatrix = SelectKBest(chi2 , k=numberOfComponents ).fit_transform(DTM, Y)

	return reducedMatrix