from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import tile
from numpy import mean
from numpy import shape

def toTFIDF(contents):
	vectorizer = CountVectorizer(min_df=1)
	X_ = vectorizer.fit_transform(contents) # under Hashed form
	X=X_.toarray()	# under array form  : Document Word Matrix	
	transformer = TfidfTransformer()
	Tfidf = transformer.fit_transform(X)
	T=Tfidf.toarray()
	return T #- tile(mean(T,1),(shape(T)[1],1)).transpose() 