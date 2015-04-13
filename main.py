import random as rnd
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import math
from sklearn.feature_extraction.text import TfidfTransformer
from tfidf import tfidf
#from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing



## 1. Load data and use preprocessing
lines = [line.strip() for line in open('r8_train_stemmed.txt')]
labels = []
contents = []
for line in lines:
    [label, content] = line.split('\t')
    labels.append(label)
    #contents.append(content.split(' '))
    contents.append(content)
N = len(lines)
#print shape(contents)
vectorizer = CountVectorizer(min_df=1)
X_ = vectorizer.fit_transform(contents) # under Hashed form
X=X_.toarray()	# under array form : Document Word Matrix

# Perform dimensionality reduction over X (SVD)
# print shape(X)
disp('Preprocessing...')

transformer = TfidfTransformer()
Tfidf = transformer.fit_transform(X)
T=Tfidf.toarray() # TF IDF matrix

#print shape(T)
#print shape(tile(mean(T,1),(shape(T)[1],1)).transpose())
T = T - tile(mean(T,1),(shape(T)[1],1)).transpose() # center matrix

U, s, V = linalg.svd(T)
U200=U[:,0:200]
##V200=V[0:200,:]
##S200=diag(s)[:200,:200]
##X200=dot(dot(U200,S200),V200)
#
X=dot(T,U)

# # samples in column, features in lines
Xtrain = X
le = preprocessing.LabelEncoder()
le.fit(labels)
Ytrain=le.transform(labels) 

savetxt('Xtrain.txt',Xtrain)
savetxt('Ytrain.txt',Ytrain)

## Read from text
#lines = [line.strip() for line in open('Xtrain.txt')]
#Xtrain = []
#for line in lines:
#    Xtrain.append([float(x) for x in line.split()])
#
#lines = [line.strip() for line in open('Ytrain.txt')]
#L=shape(lines)
#Ytrain=ones(L[0])
#i=0
#for line in lines:
#    Ytrain[i]=line
#    Ytrain[i]=int(Ytrain[i])
#    i=i+1
#
#print shape(Ytrain)
#print shape(Xtrain)
#permutation random des indices
    
#s=list(range(L[0]))
#p=random.shuffle(s)
#for i in range(L[0])    
#    Ytrain[i]=p[i]




#j = subsample_indices(x, 2)

#print [x[t] for t in j[-1]]
#print [x[t] for t in j[1]]

 # 2. Learning and Cross validation 
disp('Cross Validation...')
#n_test=10
#n_cross=10
#c=linspace(1,n_cross,n_cross)
#Gamma=linspace(1,n_cross,n_cross)
#error=zeros(n_cross*n_cross)
# iris = datasets.load_iris()
# iris.data.shape, iris.target.shape
#erreur = inf
#for i in range(n_cross):
#    for j in range(n_cross):
#        for k in range(n_test):
#         print k   
#         X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xtrain, Ytrain, test_size=(k+1)/float(n_test+1), random_state=0)
#         clf = svm.SVC(kernel='linear', C=c[i],gamma=Gamma[j]).fit(X_train, y_train)
#         error_ = clf.score(X_test,y_test)            
#         if error_ < erreur:
#          erreur = error_
#          error[i+n_cross*j]=error_
#
#index_min=argmin(error)
#C=c[index_min % n_cross]
#gamma = Gamma[( index_min - C)/n_cross]

# # 3. Retrain on full train set
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xtrain, Ytrain, test_size=(i+1)/float(n_test), random_state=0)
#clf = svm.SVC(kernel='linear', C=1).fit(Xtrain, Ytrain)


# Evaluation : Test on real test set

## Read and preprocess test data ##############################
#disp('Preprocessing on test set')
#lines = [line.strip() for line in open('r8_test_stemmed.txt')]
#labels = []
#contents = []
#for line in lines:
#    [label1,content1]=line.split(' ',1)
#    labels.append(label1)
#    contents.append(content1)
#N = len(lines)
#
#print shape(contents)
#print shape(labels)
#vectorizer = CountVectorizer(min_df=1)
#X_ = vectorizer.fit_transform(contents) # under Hashed form
#X=X_.toarray()	# under array form  : Document Word Matrix
#
#transformer = TfidfTransformer()
#Tfidf = transformer.fit_transform(X)
#T=Tfidf.toarray() # TF IDF matrix
#T=T-tile(mean(T,1), (1, shape(T)[2]))
#
#U, s, V = linalg.svd(T)
#U200=U[:,0:200]
#V200=V[0:200,:]
##S200=diag(s)[:200,:200]
##X200=dot(dot(U200,S200),V200)
##X=X200
#
#X = dot(T,U200)

# # samples in column, features in lines
#Xtest = X
#le = preprocessing.LabelEncoder()
#le.fit(labels)
#Ytest=le.transform(labels)
#
#savetxt('Xtest.txt',Xtest)
#savetxt('Ytest.txt',Ytest) 
#
### Read from text 
#lines = [line.strip() for line in open('Xtest.txt')]
#Xtest = []
#for line in lines:
#    Xtest.append([float(x) for x in line.split()])
#
#lines = [line.strip() for line in open('Ytest.txt')]
#L=shape(lines)
#Ytest=ones(L[0])
#i=0
#for line in lines:
#    Ytest[i]=line
#    Ytest[i]=int(Ytest[i])
#    i=i+1
#
#print shape(Ytest)
#print shape(Xtest)
## ###################################################
##
##error_ = clf.score(Xtest,Ytest)
##print error_
##
##
##
