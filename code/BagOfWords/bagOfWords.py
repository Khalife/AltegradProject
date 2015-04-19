#bagOfWords method
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import math
from time import time
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from toTFIDF import toTFIDF
from transformLabels import transformLabels
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


#########################################################
# 1- 
disp('Preprocessing')
with open('../data/r8_train_stemmed.txt') as lines:
    labels = []
    contents = []
    for line in lines:
        [label, content] = line.split('\t')
        labels.append(label)
        contents.append(content)
print labels[0]

T=toTFIDF(contents)
print shape(T)
svd = TruncatedSVD(n_components=100, random_state=42)
print 'PCA...'
svd.fit(T) 
X_train = svd.transform(T)
Y_train = transformLabels(labels)

# ########################################################
# # 2. Learning and Cross validation 
# # disp('Cross Validation...')
n_test=10
n_cross=10
c=linspace(1,n_cross,n_cross)
Gamma=linspace(1,n_cross,n_cross)
error=zeros(n_cross*n_cross)

for i in range(n_cross):
    print i
    for j in range(n_cross):
        for k in range(n_test):
            X_train1, X_test1, y_train1, y_test1 = cross_validation.train_test_split(X_train, Y_train, test_size=(k+1)/float(n_test+1), random_state=0)
            clf = svm.SVC(kernel='rbf', C=c[i],gamma=Gamma[j]).fit(X_train1, y_train1)
            error_ = clf.score(X_test1,y_test1)            
            error[i+n_cross*j]= error[i+n_cross*j] + error_
    	error[i+n_cross*j]=error[i+n_cross*j] / n_test # take mean 

index_min=argmin(-error)
index1 = index_min % n_cross
C_=c[index1]
gamma_ = Gamma[( index_min - index1)/n_cross]

clf = svm.SVC(kernel='rbf', C=C_, gamma = gamma_).fit(X_train, Y_train)
#clf = svm.SVC(kernel='rbf',C=1).fit(X_train,Y_train)

# ########################################################
# # 3.  
disp('Preprocessing on test set')
with open('../data/r8_test_stemmed.txt') as lines:
    labels = []
    contents = []
    for line in lines:
        [label,content]=line.split(' ',1)
        labels.append(label)
        contents.append(content)

T=toTFIDF(contents)
svd = TruncatedSVD(n_components=100, random_state=42)
print 'PCA test...'
svd.fit(T) 
X_test = svd.transform(T)
Y_test = transformLabels(labels)
<<<<<<< HEAD
print labels[0]
print labels[1]
error_ = clf.score(X_test,Y_test)
print error_
=======

Y_pred = clf.predict(X_test)

print metrics.precision_score(Y_test, Y_pred, average='micro')
print metrics.precision_score(Y_test, Y_pred, average='macro')
print metrics.recall_score(Y_test, Y_pred, average='micro')
print metrics.recall_score(Y_test, Y_pred, average='macro')
>>>>>>> 42c3737b944aa73f33fd7b42cc6e5864a9465dd4




